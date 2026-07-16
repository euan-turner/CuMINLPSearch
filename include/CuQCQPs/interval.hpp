#pragma once

#include <cstddef>
#include <vector>

namespace cuqcqps::interval
{

// An interval is the explicit form of the bounds on an
// n-dimensional region. For each dimension, we have an upperconstant memory
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

/**
 * @brief Host-side utility for representing an interval.
 *        Device-side just uses an array
 * 
 * @tparam T 
 */
template<typename T>
struct Interval
{
  std::vector<Bounds<T>> bounds;

  Interval() = default;
};

template<typename T>
struct Point
{
  std::vector<T> elems;
};

}  // namespace cuqcqps::interval
