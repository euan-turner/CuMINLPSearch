#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuinterval/cuinterval.h>

#include <cub/cub.cuh>
#include <cuda/std/limits>

#include "cuda_utils.cuh"
#include "partition.cuh"

namespace cuminlp::rosenbrock
{

// N-dimensional Rosenbrock function is
// sum_{i=1}^{N-1} (100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2)
// s.t. -30 <= x_i <= 30

// rosenbrock_kernel and interval_rosenbrock_kernel evaluate rosenbrock on
// points/intervals in parallel for testing but require explicit intervals to be
// copied over, so only work for testing on small problems.

// Rosenbrock function evaluated across a batch of points.
// ps is a flattened, row-major [N x dims] array (point i's coordinates occupy
// ps[i*dims .. i*dims+dims)).
template<typename T>
__global__ void rosenbrock_kernel(T* ps,
                                  T* out,
                                  std::size_t N,
                                  std::size_t dims)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }

  // thread evaluates out[tid] = rosenbrock(ps[tid])
  T* p = ps + tid * dims;

  T res = 0;
  for (std::size_t i = 0; i < dims - 1; ++i) {
    T xi = p[i];
    T xi1 = p[i + 1];
    T a = xi * xi - xi1;
    T b = xi - 1;
    res += 100 * a * a + b * b;
  }
  out[tid] = res;
}

template<typename T>
std::vector<T> launch_rosenbrock(const std::vector<std::vector<T>>& points)
{
  std::size_t N = points.size();
  std::size_t dims = N > 0 ? points[0].size() : 0;

  // Flatten the per-point vectors into one contiguous host buffer so a
  // single cudaMemcpy can move the whole batch.
  std::vector<T> flat(N * dims);
  for (std::size_t i = 0; i < N; ++i) {
    std::copy(points[i].begin(), points[i].end(), flat.begin() + i * dims);
  }

  dim3 BLOCK_DIM(512);
  dim3 GRID_DIM(static_cast<unsigned int>(detail::ceil_div(N, 512)));

  T* d_points = detail::alloc_device<T>(flat.size());
  T* out = detail::alloc_device<T>(N);

  detail::check(cudaMemcpy(d_points,
                           flat.data(),
                           flat.size() * sizeof(T),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy");

  rosenbrock_kernel<T><<<GRID_DIM, BLOCK_DIM>>>(d_points, out, N, dims);
  detail::check(cudaGetLastError(), "cudaGetLastError");
  detail::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<T> res(N);
  detail::check(
      cudaMemcpy(res.data(), out, N * sizeof(T), cudaMemcpyDeviceToHost),
      "cudaMemcpy");

  detail::check(cudaFree(d_points), "cudaFree");
  detail::check(cudaFree(out), "cudaFree");

  return res;
}

// Rosenbrock function evaluated across a batch of intervals.
// is is a flattened, row-major [N x dims] array of cu::interval<T>.
template<typename T>
__global__ void interval_rosenbrock_kernel(cu::interval<T>* is,
                                           cu::interval<T>* out,
                                           std::size_t N,
                                           std::size_t dims)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }

  cu::interval<T>* iv = is + tid * dims;

  cu::interval<T> res(0);
  for (std::size_t i = 0; i < dims - 1; ++i) {
    cu::interval<T> xi = iv[i];
    cu::interval<T> xi1 = iv[i + 1];

    auto a = sqr(xi) - xi1;
    auto b = xi - static_cast<T>(1);
    auto c = sqr(a) * static_cast<T>(100);
    auto d = sqr(b);
    auto e = c + d;
    res += e;
  }
  out[tid] = res;
}

template<typename T>
std::vector<cu::interval<T>> launch_interval_rosenbrock(
    const std::vector<std::vector<cu::interval<T>>>& intervals)
{
  std::size_t N = intervals.size();
  std::size_t dims = N > 0 ? intervals[0].size() : 0;

  std::vector<cu::interval<T>> flat(N * dims);
  for (std::size_t i = 0; i < N; ++i) {
    std::copy(
        intervals[i].begin(), intervals[i].end(), flat.begin() + i * dims);
  }

  dim3 BLOCK_DIM(512);
  dim3 GRID_DIM(static_cast<unsigned int>(detail::ceil_div(N, 512)));

  cu::interval<T>* d_intervals =
      detail::alloc_device<cu::interval<T>>(flat.size());
  cu::interval<T>* out = detail::alloc_device<cu::interval<T>>(N);

  detail::check(cudaMemcpy(d_intervals,
                           flat.data(),
                           flat.size() * sizeof(cu::interval<T>),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy");

  interval_rosenbrock_kernel<<<GRID_DIM, BLOCK_DIM>>>(
      d_intervals, out, N, dims);
  detail::check(cudaGetLastError(), "cudaGetLastError");
  detail::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<cu::interval<T>> res(N);
  detail::check(
      cudaMemcpy(
          res.data(), out, N * sizeof(cu::interval<T>), cudaMemcpyDeviceToHost),
      "cudaMemcpy");

  detail::check(cudaFree(d_intervals), "cudaFree");
  detail::check(cudaFree(out), "cudaFree");

  return res;
}

// CPU:

// Rosenbrock function evaluated on a single point (for correctness checks)
template<typename T>
T rosenbrock(const std::vector<T>& p)
{
  std::size_t dims = p.size();
  T res = 0;
  for (std::size_t i = 0; i < dims - 1; ++i) {
    T xi = p[i];
    T xi1 = p[i + 1];
    T a = xi * xi - xi1;
    T b = xi - 1;
    res += 100 * a * a + b * b;
  }
  return res;
}

// Partitioning and Sampling

template<std::size_t Base, std::size_t Exp>
__host__ __device__ constexpr std::size_t ipow()
{
  if constexpr (Exp == 0) {
    return 1;
  } else {
    return Base * ipow<Base, Exp - 1>();
  }
}

// One thread per (region, sample point) pair -- SamplePoints times the
// parallelism of one thread per region looping over its SamplePoints
// internally. Writes each thread's own sample value out to per_sample_ub
// rather than reducing in-kernel; reduce_sample_ub_kernel below folds each
// region's SamplePoints entries down to one upper bound afterward.
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints>
__global__ void sample_rosenbrock_kernel(
    cu::interval<T>* d_interval,
    T* per_sample_ub,  // num_regions * SamplePoints
    std::size_t dims,
    std::size_t cycle_start)
{
  constexpr std::size_t num_regions = ipow<PartitionNum, CycleSize>();
  std::size_t const gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gtid >= num_regions * SamplePoints) {
    return;
  }

  std::size_t const region = gtid / SamplePoints;
  std::size_t const i = gtid % SamplePoints;  // this thread's sample index

  partition::CycleContext<CycleSize> ctx =
      partition::make_cycle_context<CycleSize, PartitionNum>(region,
                                                             cycle_start);

  // Sample point is defined as:
  // xj = lb + i * (ub - lb) / (SamplePoints - 1) for each dimension
  // but represented as a degenerate interval [xj, xj]
  cu::interval<T> res(0);
  for (std::size_t j = 0; j < dims - 1; ++j) {
    // interval bounds on xj and xj1
    cu::interval<T> int_xj;
    cu::interval<T> int_xj1;

    partition::get_bounds(ctx, d_interval, j, int_xj);
    partition::get_bounds(ctx, d_interval, j + 1, int_xj1);

    // point bounds for this sample point
    cu::interval<T> xj(int_xj.lb
                       + i * (int_xj.ub - int_xj.lb) / (SamplePoints - 1));
    cu::interval<T> xj1(int_xj1.lb
                        + i * (int_xj1.ub - int_xj1.lb) / (SamplePoints - 1));

    auto a = sqr(xj) - xj1;
    auto b = xj - static_cast<T>(1);
    auto c = sqr(a) * static_cast<T>(100);
    auto d = sqr(b);
    auto e = c + d;
    res += e;
  }

  per_sample_ub[gtid] = res.ub;
}

// Folds the SamplePoints per-sample upper bounds sample_rosenbrock_kernel
// wrote for each region down to one upper bound per region. Kept as its own
// kernel (rather than inside sample_rosenbrock_kernel) so it's the only part
// of this pipeline still one-thread-per-region -- this part is cheap (a
// SamplePoints-long min over already-computed values, not a full Rosenbrock
// evaluation), unlike the kernel above.
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints>
__global__ void reduce_sample_ub_kernel(const T* __restrict__ per_sample_ub,
                                        T* __restrict__ thread_ubs)
{
  constexpr std::size_t num_regions = ipow<PartitionNum, CycleSize>();
  std::size_t const region = blockIdx.x * blockDim.x + threadIdx.x;
  if (region >= num_regions) {
    return;
  }

  T ub = ::cuda::std::numeric_limits<T>::max();
  for (std::size_t i = 0; i < SamplePoints; ++i) {
    ub = min(ub, per_sample_ub[region * SamplePoints + i]);
  }

  // each region has an upper bound on the minimum of the Rosenbrock function
  // over its sampled points
  thread_ubs[region] = ub;
}

// Sampling kernel returns a local upper bound on the minimum from all sampled
// points

/**
 * @brief Queries the scratch-storage size CUB's DeviceReduce::Min needs to
 * reduce `num_threads` elements. num_threads is fixed for the lifetime of a
 * driver (it's PartitionNum^CycleSize), so this is meant to be called once at
 * setup time to size a scratch buffer that's then reused every iteration --
 * never inside the per-iteration launch path.
 */
template<typename T>
std::size_t sample_rosenbrock_temp_storage_bytes(std::size_t num_threads)
{
  std::size_t temp_bytes = 0;
  detail::check(cub::DeviceReduce::Min(nullptr,
                                       temp_bytes,
                                       static_cast<T*>(nullptr),
                                       static_cast<T*>(nullptr),
                                       num_threads),
                "cub::DeviceReduce::Min (size query)");
  return temp_bytes;
}

/**
 * @brief Kernel evaluating the Rosenbrock over sampled points from each
 * subinterval. The number of subintervals is PartitionNum ^ CycleSize, with one
 * thread launched per interval.
 *
 * @tparam T Precision (float/double)
 * @tparam CycleSize Number of dimensions to partition
 * @tparam PartitionNum Number of partitions per dimension
 * @tparam SamplePoints Number of points to sample per subinterval
 * @param interval Parent interval to partition
 * @param cycle_start First dimension being partitioned
 * @param d_interval Device scratch, capacity >= interval.size(); populated
 *        here via H2D copy (shared with launch_bound_rosenbrock -- that call
 *        reuses this buffer without re-copying, since it runs immediately
 *        after with the same box)
 * @param d_thread_ubs Device scratch, capacity == num_threads
 * @param num_threads Region count this call's buffers were allocated for;
 *        must equal PartitionNum^CycleSize -- passed explicitly (rather than
 *        recomputed locally) so a caller that allocates d_thread_ubs at the
 *        wrong size is caught here instead of silently overrunning it
 * @param d_lub Device scratch, single T
 * @param d_temp_storage CUB scratch sized by
 * sample_rosenbrock_temp_storage_bytes
 * @param temp_storage_bytes size of d_temp_storage, in bytes
 * @param stream Stream every op here is enqueued on; synchronised once at the
 *        end so the returned lub is valid on return
 * @return T
 */
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints>
T launch_sample_rosenbrock(const std::vector<cu::interval<T>>& interval,
                           std::size_t cycle_start,
                           cu::interval<T>* d_interval,
                           T* d_per_sample_ub,
                           T* d_thread_ubs,
                           std::size_t num_threads,
                           T* d_lub,
                           void* d_temp_storage,
                           std::size_t temp_storage_bytes,
                           cudaStream_t stream)
{
  constexpr std::size_t expected_threads = ipow<PartitionNum, CycleSize>();
  if (num_threads != expected_threads) {
    throw std::runtime_error("launch_sample_rosenbrock: num_threads does not match "
                             "PartitionNum^CycleSize; d_thread_ubs was allocated for the "
                             "wrong number of regions");
  }

  std::size_t dims = interval.size();

  detail::check(cudaMemcpyAsync(d_interval,
                                interval.data(),
                                dims * sizeof(cu::interval<T>),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync");

  dim3 BLOCK_DIM(512);
  std::size_t const total_samples = num_threads * SamplePoints;
  dim3 SAMPLE_GRID_DIM(
      static_cast<unsigned int>(detail::ceil_div(total_samples, 512)));
  sample_rosenbrock_kernel<T, CycleSize, PartitionNum, SamplePoints>
      <<<SAMPLE_GRID_DIM, BLOCK_DIM, 0, stream>>>(
          d_interval, d_per_sample_ub, dims, cycle_start);
  detail::check(cudaGetLastError(), "cudaGetLastError");

  dim3 REDUCE_GRID_DIM(
      static_cast<unsigned int>(detail::ceil_div(num_threads, 512)));
  reduce_sample_ub_kernel<T, CycleSize, PartitionNum, SamplePoints>
      <<<REDUCE_GRID_DIM, BLOCK_DIM, 0, stream>>>(d_per_sample_ub,
                                                  d_thread_ubs);
  detail::check(cudaGetLastError(), "cudaGetLastError");

  detail::check(cub::DeviceReduce::Min(d_temp_storage,
                                       temp_storage_bytes,
                                       d_thread_ubs,
                                       d_lub,
                                       num_threads,
                                       stream),
                "cub::DeviceReduce::Min");

  T lub;
  detail::check(
      cudaMemcpyAsync(&lub, d_lub, sizeof(T), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync");
  detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  return lub;
}

// Convenience overload: allocates and frees its own device scratch, for
// callers that only run it a handful of times (tests, correctness checks) and
// so don't care about the per-call malloc/free cost. The per-iteration hot
// path (FixedRosenbrockDriver) uses the buffer-taking overload above instead,
// with everything pre-allocated once outside its loop.
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints>
T launch_sample_rosenbrock(const std::vector<cu::interval<T>>& interval,
                           std::size_t cycle_start)
{
  std::size_t const num_threads = ipow<PartitionNum, CycleSize>();

  cu::interval<T>* d_interval =
      detail::alloc_device<cu::interval<T>>(interval.size());
  T* d_per_sample_ub = detail::alloc_device<T>(num_threads * SamplePoints);
  T* d_thread_ubs = detail::alloc_device<T>(num_threads);
  T* d_lub = detail::alloc_device<T>(1);
  std::size_t const temp_storage_bytes =
      sample_rosenbrock_temp_storage_bytes<T>(num_threads);
  unsigned char* d_temp_storage =
      detail::alloc_device<unsigned char>(temp_storage_bytes);

  cudaStream_t stream;
  detail::check(cudaStreamCreate(&stream), "cudaStreamCreate");

  T const lub =
      launch_sample_rosenbrock<T, CycleSize, PartitionNum, SamplePoints>(
          interval,
          cycle_start,
          d_interval,
          d_per_sample_ub,
          d_thread_ubs,
          num_threads,
          d_lub,
          d_temp_storage,
          temp_storage_bytes,
          stream);

  detail::check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  detail::check(cudaFree(d_interval), "cudaFree");
  detail::check(cudaFree(d_per_sample_ub), "cudaFree");
  detail::check(cudaFree(d_thread_ubs), "cudaFree");
  detail::check(cudaFree(d_lub), "cudaFree");
  detail::check(cudaFree(d_temp_storage), "cudaFree");

  return lub;
}

template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
__global__ void bound_rosenbrock_kernel(cu::interval<T>* d_interval,
                                        T gub,
                                        bool* d_prune_interval,
                                        T* d_interval_lb,
                                        std::size_t dims,
                                        std::size_t cycle_start)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= ipow<PartitionNum, CycleSize>()) {
    return;
  }

  partition::CycleContext<CycleSize> ctx =
      partition::make_cycle_context<CycleSize, PartitionNum>(tid, cycle_start);

  // Evaluate rosenbrock kernel over the current interval
  cu::interval<T> res(0);
  for (std::size_t i = 0; i < dims - 1; ++i) {
    // interval bounds on xi and xi1
    cu::interval<T> xi;
    cu::interval<T> xi1;

    partition::get_bounds(ctx, d_interval, i, xi);
    partition::get_bounds(ctx, d_interval, i + 1, xi1);

    auto a = sqr(xi) - xi1;
    auto b = xi - static_cast<T>(1);
    auto c = sqr(a) * static_cast<T>(100);
    auto d = sqr(b);
    auto e = c + d;
    res += e;
  }
  // is region suboptimal?
  d_prune_interval[tid] = (res.lb > gub);
  d_interval_lb[tid] = res.lb;
}

// d_interval is not re-copied here: it's the same buffer
// launch_sample_rosenbrock just populated
template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
void launch_bound_rosenbrock(
    std::size_t dims,
    T gub,
    std::size_t cycle_start,
    cu::interval<T>* d_interval,
    bool* d_prune_interval,
    T* d_interval_lb,
    std::size_t num_threads,
    std::span<bool, ipow<PartitionNum, CycleSize>()> interval_results,
    std::span<T, ipow<PartitionNum, CycleSize>()> lb_results,
    cudaStream_t stream)
{
  constexpr std::size_t expected_threads = ipow<PartitionNum, CycleSize>();
  if (num_threads != expected_threads) {
    throw std::runtime_error("launch_bound_rosenbrock: num_threads does not match "
                             "PartitionNum^CycleSize; d_prune_interval/d_interval_lb were "
                             "allocated for the wrong number of regions");
  }

  dim3 BLOCK_DIM(512);
  dim3 GRID_DIM(static_cast<unsigned int>(detail::ceil_div(num_threads, 512)));
  bound_rosenbrock_kernel<T, CycleSize, PartitionNum>
      <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
          d_interval, gub, d_prune_interval, d_interval_lb, dims, cycle_start);
  detail::check(cudaGetLastError(), "cudaGetLastError");

  // populates an array of PartitionNum ^ CycleSize bools (one per interval)
  detail::check(cudaMemcpyAsync(interval_results.data(),
                                d_prune_interval,
                                num_threads * sizeof(bool),
                                cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync");
  detail::check(cudaMemcpyAsync(lb_results.data(),
                                d_interval_lb,
                                num_threads * sizeof(T),
                                cudaMemcpyDeviceToHost,
                                stream),
                "cudaMemcpyAsync");

  detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

// Convenience overload: allocates and frees its own device scratch (see
// launch_sample_rosenbrock's convenience overload above for why this exists
// alongside the buffer-taking core).
template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
void launch_bound_rosenbrock(
    const std::vector<cu::interval<T>>& interval,
    T gub,
    std::size_t cycle_start,
    std::span<bool, ipow<PartitionNum, CycleSize>()> interval_results,
    std::span<T, ipow<PartitionNum, CycleSize>()> lb_results)
{
  std::size_t const num_threads = ipow<PartitionNum, CycleSize>();

  cu::interval<T>* d_interval =
      detail::alloc_device<cu::interval<T>>(interval.size());
  bool* d_prune_interval = detail::alloc_device<bool>(num_threads);
  T* d_interval_lb = detail::alloc_device<T>(num_threads);

  cudaStream_t stream;
  detail::check(cudaStreamCreate(&stream), "cudaStreamCreate");

  detail::check(cudaMemcpyAsync(d_interval,
                                interval.data(),
                                interval.size() * sizeof(cu::interval<T>),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync");
  detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  launch_bound_rosenbrock<T, CycleSize, PartitionNum>(interval.size(),
                                                      gub,
                                                      cycle_start,
                                                      d_interval,
                                                      d_prune_interval,
                                                      d_interval_lb,
                                                      num_threads,
                                                      interval_results,
                                                      lb_results,
                                                      stream);

  detail::check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  detail::check(cudaFree(d_interval), "cudaFree");
  detail::check(cudaFree(d_prune_interval), "cudaFree");
  detail::check(cudaFree(d_interval_lb), "cudaFree");
}

}  // namespace cuminlp::rosenbrock
