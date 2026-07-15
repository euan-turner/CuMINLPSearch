#pragma once

#include "interval.hpp"

namespace interval
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

template<typename T>
__device__ inline Bounds<T> add_bounds(Bounds<T> a, Bounds<T> b)
{
  return Bounds<T>(add_rd(a.lower, b.lower), add_ru(a.upper, b.upper));
}

template<typename T>
__device__ inline Bounds<T> sub_bounds(Bounds<T> a, Bounds<T> b)
{
  return Bounds<T>(sub_rd(a.lower, b.upper), sub_ru(a.upper, b.lower));
}

template<typename T>
__device__ inline Bounds<T> mul_bounds(Bounds<T> a, Bounds<T> b)
{
  Bounds<T> c;

  c.lower = fmin(fmin(mul_rd(a.lower, b.lower), mul_rd(a.lower, b.upper)),
                 fmin(mul_rd(a.upper, b.lower), mul_rd(a.upper, b.upper)));
  c.upper = fmax(fmax(mul_ru(a.lower, b.lower), mul_ru(a.lower, b.upper)),
                 fmax(mul_ru(a.upper, b.lower), mul_ru(a.upper, b.upper)));

  return c;
}

template<typename T>
__device__ inline Bounds<T> sqr_bound(Bounds<T> a)
{
  if (a.lower >= 0) {
    return Bounds<T>(mul_rd(a.lower, a.lower), mul_ru(a.upper, a.upper));
  } else if (a.upper <= 0) {
    return Bounds<T>(mul_rd(a.upper, a.upper), mul_ru(a.lower, a.lower));
  } else {
    return Bounds<T>(0,
                     fmax(mul_ru(a.lower, a.lower), mul_ru(a.upper, a.upper)));
  }
}

template<typename T>
__device__ inline Bounds<T> scal_add_bound(Bounds<T> a, T x)
{
  return Bounds<T>(add_rd(a.lower, x), add_ru(a.upper, x));
}

template<typename T>
__device__ inline Bounds<T> scal_sub_bound(Bounds<T> a, T x)
{
  return Bounds<T>(sub_rd(a.lower, x), sub_ru(a.upper, x));
}

template<typename T>
__device__ inline Bounds<T> scal_sub_bound(T x, Bounds<T> a)
{
  return Bounds<T>(sub_rd(x, a.upper), sub_ru(x, a.lower));
}

template<typename T>
__device__ inline Bounds<T> scal_mul_bound(Bounds<T> a, T x)
{
  if (x >= 0) {
    return Bounds<T>(mul_rd(a.lower, x), mul_ru(a.upper, x));
  } else {
    return Bounds<T>(mul_rd(a.upper, x), mul_ru(a.lower, x));
  }
}

// TODO: make these operators on the interval struct

// elementwise square of each dimension bound for the interval
template<typename T, std::size_t N>
__device__ inline Interval<T, N> sqr(Interval<T, N> a)
{
  Interval<T, N> b;
  for (std::size_t i = 0; i < N; ++i) {
    b.bounds[i] = sqr_bound<T>(a.bounds[i]);
  }
  return b;
}

// elementwise scale by a parallel real coefficient vector, e.g. diag(Q) or b
// in x^T Q x + b^T x
template<typename T, std::size_t N>
__device__ inline Interval<T, N> scale(Interval<T, N> a, const T (&coeffs)[N])
{
  Interval<T, N> b;
  for (std::size_t i = 0; i < N; ++i) {
    b.bounds[i] = scal_mul_bound(a.bounds[i], coeffs[i]);
  }
  return b;
}

// horizontal fold of a per-dimension interval vector down to a single bound,
// the step every per-dimension term (diagonal, linear) needs before it can
// be added into the objective
template<typename T, std::size_t N>
__device__ inline Bounds<T> reduce_sum(Interval<T, N> a)
{
  Bounds<T> res(0);
  for (std::size_t i = 0; i < N; ++i) {
    res = add_bounds(res, a.bounds[i]);
  }
  return res;
}

// to evaluate x^T Q x + b^T x + c
// reduce_sum(scale(sqr(x), diag(Q)))
// reduce_sum(scale(x, b))
// loop i j, mul_bounds, scal_mul_bound for non-diagonal Q terms

}  // namespace interval
