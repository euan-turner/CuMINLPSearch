#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda/std/limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuinterval.cuh"
#include "opt.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t cuda_check_err_ = (expr); \
    if (cuda_check_err_ != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA error: ") \
                                + cudaGetErrorString(cuda_check_err_) + " (" #expr ") at " __FILE__ \
                                ":" + std::to_string(__LINE__)); \
    } \
  } while (false)

// N-dimensional Rosenbrock function is
// sum_{i=1}^{N-1} (100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2)
// s.t. -30 <= x_i <= 30

// rosenbrock_kernel and interval_rosenbrock_kernel evaluate rosenbrock on points/intervals in parallel for testing
// but require explicit intervals to be copied over, so only work for testing on small problems.

// Rosenbrock function evaluated across a batch of points.
// ps is a flattened, row-major [N x dims] array (point i's coordinates occupy
// ps[i*dims .. i*dims+dims)).
template<typename T>
__global__ void rosenbrock_kernel(T *ps, T *out, std::size_t N, std::size_t dims) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  // thread evaluates out[tid] = rosenbrock(ps[tid])
  T *p = ps + tid * dims;

  T res = 0;
  for (std::size_t i = 0; i < dims - 1; ++i) {
    T xi = p[i];
    T xi1 = p[i+1];
    T a = xi * xi - xi1;
    T b = xi - 1;
    res += 100 * a * a + b * b;
  }
  out[tid] = res;
}


template<typename T>
std::vector<T> launch_rosenbrock(const std::vector<interval::Point<T>> &points) {
  std::size_t N = points.size();
  std::size_t dims = N > 0 ? points[0].elems.size() : 0;

  // Flatten the per-point vectors into one contiguous host buffer so a
  // single cudaMemcpy can move the whole batch.
  std::vector<T> flat(N * dims);
  for (std::size_t i = 0; i < N; ++i) {
    std::copy(points[i].elems.begin(), points[i].elems.end(), flat.begin() + i * dims);
  }

  dim3 BLOCK_DIM(512);
  dim3 GRID_DIM(CEIL_DIV(N, 512));

  T *d_points;
  T *out;

  CUDA_CHECK(cudaMalloc(&d_points, flat.size() * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&out, N * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(d_points, flat.data(), flat.size() * sizeof(T), cudaMemcpyHostToDevice));

  rosenbrock_kernel<T><<<GRID_DIM, BLOCK_DIM>>>(d_points, out, N, dims);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<T> res(N);
  CUDA_CHECK(cudaMemcpy(res.data(), out, N * sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(out));

  return res;
}

// Rosenbrock function evaluated across a batch of intervals.
// is is a flattened, row-major [N x dims] array of Bounds<T>.
template<typename T>
__global__ void interval_rosenbrock_kernel(interval::Bounds<T> *is, interval::Bounds<T> *out, std::size_t N, std::size_t dims) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  interval::Bounds<T> *iv = is + tid * dims;

  interval::Bounds<T> res(0);
  for (std::size_t i = 0; i < dims - 1; ++i) {
    interval::Bounds<T> xi = iv[i];
    interval::Bounds<T> xi1 = iv[i+1];

    interval::Bounds<T> a = interval::sub_bounds<T>(interval::sqr_bound<T>(xi), xi1);
    interval::Bounds<T> b = interval::scal_sub_bound<T>(xi, static_cast<T>(1));

    interval::Bounds<T> c = interval::scal_mul_bound<T>(interval::sqr_bound<T>(a), static_cast<T>(100));
    interval::Bounds<T> d = interval::sqr_bound<T>(b);
    interval::Bounds<T> e = interval::add_bounds<T>(c, d);
    res = interval::add_bounds<T>(res, e);
  }
  out[tid] = res;
}


template<typename T>
std::vector<interval::Bounds<T>> launch_interval_rosenbrock(const std::vector<interval::Interval<T>> &intervals) {
  std::size_t N = intervals.size();
  std::size_t dims = N > 0 ? intervals[0].bounds.size() : 0;

  std::vector<interval::Bounds<T>> flat(N * dims);
  for (std::size_t i = 0; i < N; ++i) {
    std::copy(intervals[i].bounds.begin(), intervals[i].bounds.end(), flat.begin() + i * dims);
  }

  dim3 BLOCK_DIM(512);
  dim3 GRID_DIM(CEIL_DIV(N, 512));

  interval::Bounds<T> *d_intervals;
  interval::Bounds<T> *out;

  CUDA_CHECK(cudaMalloc(&d_intervals, flat.size() * sizeof(interval::Bounds<T>)));
  CUDA_CHECK(cudaMalloc(&out, N * sizeof(interval::Bounds<T>)));

  CUDA_CHECK(cudaMemcpy(d_intervals, flat.data(), flat.size() * sizeof(interval::Bounds<T>), cudaMemcpyHostToDevice));

  interval_rosenbrock_kernel<<<GRID_DIM, BLOCK_DIM>>>(d_intervals, out, N, dims);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<interval::Bounds<T>> res(N);
  CUDA_CHECK(cudaMemcpy(res.data(), out, N * sizeof(interval::Bounds<T>), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_intervals));
  CUDA_CHECK(cudaFree(out));

  return res;
}

// CPU:

// Rosenbrock function evaluated on a single point (for correctness checks)
template<typename T>
T rosenbrock(const interval::Point<T> &p) {
  std::size_t dims = p.elems.size();
  T res = 0;
  for (std::size_t i = 0; i < dims - 1; ++i) {
    T xi = p.elems[i];
    T xi1 = p.elems[i+1];
    T a = xi * xi - xi1;
    T b = xi - 1;
    res += 100 * a * a + b * b;
  }
  return res;
}




// Partitioning and Sampling


template <std::size_t Base, std::size_t Exp>
__host__ __device__
constexpr std::size_t ipow() {
  if constexpr (Exp == 0) { return 1; }
  else { return Base * ipow<Base, Exp - 1>(); }
}

template<typename T, std::size_t CycleSize, std::size_t PartitionNum, std::size_t SamplePoints>
__global__ void sample_rosenbrock_kernel(interval::Bounds<T> *d_interval, T *thread_ubs, std::size_t dims, std::size_t cycle_start) 
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= ipow<PartitionNum, CycleSize>()) return;

  CycleContext<CycleSize> ctx = make_cycle_context<CycleSize, PartitionNum>(tid, cycle_start);

  T ub = ::cuda::std::numeric_limits<T>::max(); // upper bound on the minimum of the Rosenbrock function, based on these sampled points
  for (std::size_t i = 0; i < SamplePoints; ++i) {
    // Sample point is defined as:
    // xi = lb + i * (ub - lb) / SamplePoints for each dimension
    // but represented as a degenerate interval [xi, xi]


    // Evaluate Rosenbrock on the degenerate interval for xi
    interval::Bounds<T> res(0);
    for (std::size_t j = 0; j < dims - 1; ++j) {
      // interval bounds on xj and xj1
      interval::Bounds<T> int_xj;
      interval::Bounds<T> int_xj1;

      get_bounds(ctx, d_interval, j, int_xj);
      get_bounds(ctx, d_interval, j + 1, int_xj1);

      // point bounds for this sample point
      interval::Bounds<T> xj(int_xj.lower + i * (int_xj.upper - int_xj.lower) / (SamplePoints - 1));
      interval::Bounds<T> xj1(int_xj1.lower + i * (int_xj1.upper - int_xj1.lower) / (SamplePoints -1));
      
      interval::Bounds<T> a = interval::sub_bounds<T>(interval::sqr_bound<T>(xj), xj1);
      interval::Bounds<T> b = interval::scal_sub_bound<T>(xj, static_cast<T>(1));

      interval::Bounds<T> c = interval::scal_mul_bound<T>(interval::sqr_bound<T>(a), static_cast<T>(100));
      interval::Bounds<T> d = interval::sqr_bound<T>(b);
      interval::Bounds<T> e = interval::add_bounds<T>(c, d);
      res = interval::add_bounds<T>(res, e);
    }
    ub = min(ub, res.upper);

  }

  // each thread has an upper bound on the minimum of the Rosenbrock function over its sampled points
  thread_ubs[tid] = ub;
} 


// Sampling kernel returns a local upper bound on the minimum from all sampled points

/**
 * @brief Kernel evaluating the Rosenbrock over sampled points from each subinterval.
 * The number of subintervals is PartitionNum ^ CycleSize, with one thread launched per interval.
 * 
 * @tparam T Precision (float/double)
 * @tparam CycleSize Number of dimensions to partition
 * @tparam PartitionNum Number of partitions per dimension
 * @tparam SamplePoints Number of points to sample per subinterval
 * @param interval Parent interval to partition
 * @param cycle_start First dimension being partitioned
 * @return T 
 */
template<typename T, std::size_t CycleSize, std::size_t PartitionNum, std::size_t SamplePoints>
T launch_sample_rosenbrock(const interval::Interval<T>& interval, std::size_t cycle_start)
{
  interval::Bounds<T> *d_interval;
  T *d_thread_ubs;
  T *d_lub;

  std::size_t dims = interval.bounds.size();
  std::size_t num_threads = ipow<PartitionNum, CycleSize>();


  CUDA_CHECK(cudaMalloc(&d_interval, dims * sizeof(interval::Bounds<T>)));
  CUDA_CHECK(cudaMalloc(&d_thread_ubs, num_threads * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_lub, sizeof(T)));

  CUDA_CHECK(cudaMemcpy(d_interval, interval.bounds.data(), dims * sizeof(interval::Bounds<T>), cudaMemcpyHostToDevice));
  
  dim3 BLOCK_DIM(512);
  dim3 GRID_DIM(CEIL_DIV(num_threads, 512));
  sample_rosenbrock_kernel<T, CycleSize, PartitionNum, SamplePoints><<<GRID_DIM, BLOCK_DIM>>>(d_interval, d_thread_ubs, dims, cycle_start);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::size_t temp_bytes = 0;

  cub::DeviceReduce::Min(nullptr, temp_bytes, d_thread_ubs, d_lub, num_threads);

  void* d_temp_storage = nullptr;
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_bytes));

  cub::DeviceReduce::Min(d_temp_storage, temp_bytes, d_thread_ubs, d_lub, num_threads);

  T lub;
  CUDA_CHECK(cudaMemcpy(&lub, d_lub, sizeof(T), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_interval));
  CUDA_CHECK(cudaFree(d_thread_ubs));
  CUDA_CHECK(cudaFree(d_lub));
  CUDA_CHECK(cudaFree(d_temp_storage));

  return lub;
}


template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
__global__ void bound_rosenbrock_kernel(interval::Bounds<T> *d_interval, T gub, bool *d_prune_interval, std::size_t dims, std::size_t cycle_start)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= ipow<PartitionNum, CycleSize>()) return;

  CycleContext<CycleSize> ctx = make_cycle_context<CycleSize, PartitionNum>(tid, cycle_start);
  
  // Evaluate rosenbrock kernel over the current interval
  interval::Bounds<T> res(0);
  for (std::size_t i = 0; i < dims - 1; ++i) {
    // interval bounds on xi and xi1
    interval::Bounds<T> xi;
    interval::Bounds<T> xi1;

    get_bounds(ctx, d_interval, i, xi);
    get_bounds(ctx, d_interval, i + 1, xi1);

    interval::Bounds<T> a = interval::sub_bounds<T>(interval::sqr_bound<T>(xi), xi1);
    interval::Bounds<T> b = interval::scal_sub_bound<T>(xi, static_cast<T>(1));

    interval::Bounds<T> c = interval::scal_mul_bound<T>(interval::sqr_bound<T>(a), static_cast<T>(100));
    interval::Bounds<T> d = interval::sqr_bound<T>(b);
    interval::Bounds<T> e = interval::add_bounds<T>(c, d);
    res = interval::add_bounds<T>(res, e);
  }
  // is region suboptimal?
  d_prune_interval[tid] = (res.lower > gub);
}


template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
void launch_bound_rosenbrock(const interval::Interval<T>& interval, T gub, std::size_t cycle_start, std::array<bool, ipow<PartitionNum, CycleSize>()>& interval_results)
{
  interval::Bounds<T> *d_interval;
  bool *d_prune_interval; // true if region is suboptimal, or infeasible

  std::size_t dims = interval.bounds.size();
  std::size_t num_threads = ipow<PartitionNum, CycleSize>();

  CUDA_CHECK(cudaMalloc(&d_interval, dims * sizeof(interval::Bounds<T>)));
  CUDA_CHECK(cudaMalloc(&d_prune_interval, num_threads * sizeof(bool)));

  CUDA_CHECK(cudaMemcpy(d_interval, interval.bounds.data(), dims * sizeof(interval::Bounds<T>), cudaMemcpyHostToDevice));

  dim3 BLOCK_DIM(512);
  dim3 GRID_DIM(CEIL_DIV(num_threads, 512));
  bound_rosenbrock_kernel<T, CycleSize, PartitionNum><<<GRID_DIM, BLOCK_DIM>>>(d_interval, gub, d_prune_interval, dims, cycle_start);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // populates an array of PartitionNum ^ CycleSize bools (one per interval)
  CUDA_CHECK(cudaMemcpy(interval_results.data(), d_prune_interval, num_threads * sizeof(bool), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_interval));
  CUDA_CHECK(cudaFree(d_prune_interval));
}