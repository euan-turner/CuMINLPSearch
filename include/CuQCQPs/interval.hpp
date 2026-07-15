#pragma once

#include <cstddef>

namespace interval
{

// An interval is the explicit form of the bounds on an
// n-dimensional region. For each dimension, we have an upper
// and lower bound. Since most interval operations use the lower
// and upper bound of an interval together, Array of Structs is used.

template<typename T>
struct Bounds
{
  T lower;
  T upper;

  Bounds() = default;

  // point bound
  __host__ __device__ explicit constexpr Bounds(T a)
      : lower(a)
      , upper(a)
  {
  }

  __host__ __device__ constexpr Bounds(T lb, T ub)
      : lower(lb)
      , upper(ub)
  {
  }
};

template<typename T, std::size_t N>
struct Interval
{
  Bounds<T> bounds[N];

  Interval() = default;
};

template<typename T, std::size_t N>
struct Point
{
  T elems[N];
};

}  // namespace interval
