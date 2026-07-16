#pragma once

#include "interval.hpp"

namespace cuqcqps::interval
{

// Interval analysis of QCQPs requires:
// addition
// subtraction
// multiplication
// squaring

// Implementations here are for a single thread processing an entire interval
// TODO: constexpr?
__device__ inline float add_rd(float a, float b)
{
  return __fadd_rd(a, b);
}

__device__ inline double add_rd(double a, double b)
{
  return __dadd_rd(a, b);
}

__device__ inline float add_ru(float a, float b)
{
  return __fadd_ru(a, b);
}

__device__ inline double add_ru(double a, double b)
{
  return __dadd_ru(a, b);
}

__device__ inline float sub_rd(float a, float b)
{
  return __fsub_rd(a, b);
}

__device__ inline double sub_rd(double a, double b)
{
  return __dsub_rd(a, b);
}

__device__ inline float sub_ru(float a, float b)
{
  return __fsub_ru(a, b);
}

__device__ inline double sub_ru(double a, double b)
{
  return __dsub_ru(a, b);
}

__device__ inline float mul_rd(float a, float b)
{
  return __fmul_rd(a, b);
}

__device__ inline double mul_rd(double a, double b)
{
  return __dmul_rd(a, b);
}

__device__ inline float mul_ru(float a, float b)
{
  return __fmul_ru(a, b);
}

__device__ inline double mul_ru(double a, double b)
{
  return __dmul_ru(a, b);
}

// Default Rounding policy for the composed ops below; tests substitute
// CpuRounding (test/source/cpu_rounding.hpp) to run them on the host.
struct CudaRounding
{
  __device__ static inline float add_rd(float a, float b)
  {
    return interval::add_rd(a, b);
  }

  __device__ static inline double add_rd(double a, double b)
  {
    return interval::add_rd(a, b);
  }

  __device__ static inline float add_ru(float a, float b)
  {
    return interval::add_ru(a, b);
  }

  __device__ static inline double add_ru(double a, double b)
  {
    return interval::add_ru(a, b);
  }

  __device__ static inline float sub_rd(float a, float b)
  {
    return interval::sub_rd(a, b);
  }

  __device__ static inline double sub_rd(double a, double b)
  {
    return interval::sub_rd(a, b);
  }

  __device__ static inline float sub_ru(float a, float b)
  {
    return interval::sub_ru(a, b);
  }

  __device__ static inline double sub_ru(double a, double b)
  {
    return interval::sub_ru(a, b);
  }

  __device__ static inline float mul_rd(float a, float b)
  {
    return interval::mul_rd(a, b);
  }

  __device__ static inline double mul_rd(double a, double b)
  {
    return interval::mul_rd(a, b);
  }

  __device__ static inline float mul_ru(float a, float b)
  {
    return interval::mul_ru(a, b);
  }

  __device__ static inline double mul_ru(double a, double b)
  {
    return interval::mul_ru(a, b);
  }
};

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> add_bounds(Bounds<T> a, Bounds<T> b)
{
  return Bounds<T>(Rounding::add_rd(a.lower, b.lower),
                   Rounding::add_ru(a.upper, b.upper));
}

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> sub_bounds(Bounds<T> a, Bounds<T> b)
{
  return Bounds<T>(Rounding::sub_rd(a.lower, b.upper),
                   Rounding::sub_ru(a.upper, b.lower));
}

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> mul_bounds(Bounds<T> a, Bounds<T> b)
{
  Bounds<T> c;

  c.lower = fmin(fmin(Rounding::mul_rd(a.lower, b.lower),
                      Rounding::mul_rd(a.lower, b.upper)),
                 fmin(Rounding::mul_rd(a.upper, b.lower),
                      Rounding::mul_rd(a.upper, b.upper)));
  c.upper = fmax(fmax(Rounding::mul_ru(a.lower, b.lower),
                      Rounding::mul_ru(a.lower, b.upper)),
                 fmax(Rounding::mul_ru(a.upper, b.lower),
                      Rounding::mul_ru(a.upper, b.upper)));

  return c;
}

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> sqr_bound(Bounds<T> a)
{
  if (a.lower >= 0) {
    return Bounds<T>(Rounding::mul_rd(a.lower, a.lower),
                     Rounding::mul_ru(a.upper, a.upper));
  } else if (a.upper <= 0) {
    return Bounds<T>(Rounding::mul_rd(a.upper, a.upper),
                     Rounding::mul_ru(a.lower, a.lower));
  } else {
    return Bounds<T>(0,
                     fmax(Rounding::mul_ru(a.lower, a.lower),
                          Rounding::mul_ru(a.upper, a.upper)));
  }
}

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> scal_add_bound(Bounds<T> a, T x)
{
  return Bounds<T>(Rounding::add_rd(a.lower, x), Rounding::add_ru(a.upper, x));
}

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> scal_sub_bound(Bounds<T> a, T x)
{
  return Bounds<T>(Rounding::sub_rd(a.lower, x), Rounding::sub_ru(a.upper, x));
}

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> scal_sub_bound(T x, Bounds<T> a)
{
  return Bounds<T>(Rounding::sub_rd(x, a.upper), Rounding::sub_ru(x, a.lower));
}

template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> scal_mul_bound(Bounds<T> a, T x)
{
  if (x >= 0) {
    return Bounds<T>(Rounding::mul_rd(a.lower, x),
                     Rounding::mul_ru(a.upper, x));
  } else {
    return Bounds<T>(Rounding::mul_rd(a.upper, x),
                     Rounding::mul_ru(a.lower, x));
  }
}

// elementwise square of each dimension bound
template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline void sqr(const Bounds<T>* a,
                                    Bounds<T>* out,
                                    std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = sqr_bound<T, Rounding>(a[i]);
  }
}

// elementwise scale by a parallel real coefficient array, e.g. diag(Q) or b
// in x^T Q x + b^T x
template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline void scale(const Bounds<T>* a,
                                      const T* coeffs,
                                      Bounds<T>* out,
                                      std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = scal_mul_bound<T, Rounding>(a[i], coeffs[i]);
  }
}

// horizontal fold of a per-dimension bound array down to a single bound,
// the step every per-dimension term (diagonal, linear) needs before it can
// be added into the objective
template<typename T, typename Rounding = CudaRounding>
__host__ __device__ inline Bounds<T> reduce_sum(const Bounds<T>* a,
                                                std::size_t n)
{
  Bounds<T> res(0);
  for (std::size_t i = 0; i < n; ++i) {
    res = add_bounds<T, Rounding>(res, a[i]);
  }
  return res;
}

// to evaluate x^T Q x + b^T x + c, with x untouched for reuse across both
// terms: sqr(x, scratch, n); scale(scratch, diag(Q), scratch, n);
// reduce_sum(scratch, n) scale(x, b, scratch, n); reduce_sum(scratch, n) loop i
// j, mul_bounds, scal_mul_bound for non-diagonal Q terms

}  // namespace cuqcqps::interval
