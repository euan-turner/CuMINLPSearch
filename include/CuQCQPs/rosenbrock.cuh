#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuinterval.cuh"

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

// Kernel Launches:

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

  rosenbrock_kernel<<<GRID_DIM, BLOCK_DIM>>>(d_points, out, N, dims);
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

// TODO: CPU equivalent check?