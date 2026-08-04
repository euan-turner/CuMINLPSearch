#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuinterval/cuinterval.h>
#include <cuinterval/interval.h>

#include <cub/cub.cuh>
#include <cuda/std/limits>

#include "composition_policy.hpp"
#include "cuda_utils.cuh"
#include "dag.hpp"
#include "partition.cuh"
#include "search_sizing.hpp"

namespace cuminlp::dag
{

/**
 * @brief Partitions the `parent_domain` into materialised sub-domains
 *
 * @tparam T
 * @tparam CycleSize
 * @param parent_domain
 * @param slot_var_ids  which variable each of the CycleSize slots acts on
 * @param slot_fan_out  fan-out (radix) of each slot
 * @param slot_kind     operation each slot performs
 * @param var_buffers
 * @param n_vars
 * @param n_regions
 * @return __global__
 */
template<typename T, std::size_t CycleSize>
__global__ void partition_variables_kernel(
    const cu::interval<T>* __restrict__ parent_domain,  // NUM_VARS (box bounds
                                                        // per variable)
    const std::size_t* __restrict__ slot_var_ids,
    const std::uint32_t* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind,
    cu::interval<T>* const* __restrict__ var_buffers,  // NUM_VARS x NUM_REGIONS
                                                       // (box bounds per
                                                       // variable per region)
    std::size_t n_vars,
    std::size_t n_regions,
    std::size_t slot_count)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t r = tid; r < n_regions; r += num_threads) {
    auto ctx = partition::make_slot_context<CycleSize>(
        r, slot_var_ids, slot_fan_out, slot_kind, slot_count);
    for (std::size_t vid = 0; vid < n_vars; ++vid) {
      cu::interval<T> v;
      partition::get_slot_bounds(ctx, parent_domain, vid, v);
      var_buffers[vid][r] = v;
    }
  }
}

/**
 * @brief Deterministic counterpart to sample_points_kernel: every slot must
 * be IntegerEnumerate/BinaryEnumerate (GraphDriver only wires this up when
 * that holds and every live variable fits in CycleSize slots), so every
 * region's per-variable bound from get_slot_bounds is already an exact point
 * (lb == ub) -- nothing to sample, just take it. Otherwise identical to
 * partition_variables_kernel, just writing the scalar T rather than
 * cu::interval<T>, so it plugs into the same V=T op-kernel pipeline
 * (feasibility_check_kernel, objective_extract_kernel, CUB ArgMin) that
 * sample_points_kernel already feeds -- this reuses that pipeline as-is to
 * get the true minimum objective over every point in the enumeration, not
 * an interval relaxation.
 *
 * @tparam T
 * @tparam CycleSize
 * @param parent_domain
 * @param slot_var_ids  which variable each of the CycleSize slots acts on
 * @param slot_fan_out  fan-out (radix) of each slot
 * @param slot_kind     operation each slot performs (always Enumerate here)
 * @param var_buffers
 * @param n_vars
 * @param n_regions
 * @return __global__
 */
template<typename T, std::size_t CycleSize>
__global__ void enumerate_points_kernel(
    const cu::interval<T>* __restrict__ parent_domain,
    const std::size_t* __restrict__ slot_var_ids,
    const std::uint32_t* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind,
    T* const* __restrict__ var_buffers,  // NUM_VARS x NUM_REGIONS
    std::size_t n_vars,
    std::size_t n_regions,
    std::size_t slot_count)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t r = tid; r < n_regions; r += num_threads) {
    auto ctx = partition::make_slot_context<CycleSize>(
        r, slot_var_ids, slot_fan_out, slot_kind, slot_count);
    for (std::size_t vid = 0; vid < n_vars; ++vid) {
      cu::interval<T> v;
      partition::get_slot_bounds(ctx, parent_domain, vid, v);
      var_buffers[vid][r] = v.lb;  // exact point: v.lb == v.ub
    }
  }
}

// Combines the four coordinates with distinct odd multipliers, then runs the
// SplitMix64 finaliser over the result to avalanche them together. Shared by
// sample_from_interval (continuous) and sample_discrete_from_interval
// (integer/binary) below, so both draw from the same seed stream.
__device__ __forceinline__ std::uint64_t sample_hash(std::size_t vid,
                                                     std::size_t i,
                                                     std::size_t region,
                                                     std::size_t salt)
{
  std::uint64_t x = (std::uint64_t)vid * 0x9e3779b97f4a7c15ULL;
  x ^= (std::uint64_t)i * 0xbf58476d1ce4e5b9ULL;
  x ^= (std::uint64_t)region * 0x94d049bb133111ebULL;
  x ^= (std::uint64_t)salt * 0xd6e8feb86659fd93ULL;

  // SplitMix64
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

template<typename T>
__device__ __forceinline__ T sample_from_interval(T lb,
                                                  T ub,
                                                  std::size_t vid,
                                                  std::size_t i,
                                                  std::size_t region,
                                                  std::size_t salt)
{
  std::uint64_t x = sample_hash(vid, i, region, salt);

  T u;
  if constexpr (std::is_same_v<T, float>) {
    u = (uint32_t)x * (T(1.0) / T(4294967296.0));
  } else {
    u = (x >> 11) * (T(1.0) / T(9007199254740992.0));
  }
  return fma(u, ub - lb, lb);
}

// Uniformly samples one of the integers in [lb, ub] rather than a continuous
// value -- for Integer/Binary variables, a fractional "sample" isn't a
// feasible point of the original problem, so it can never be a valid GUB
// witness. ceil/floor (rather than assuming lb/ub are themselves already
// integers) is what keeps this correct even for a sub-box produced by
// bisecting an integer variable whose domain was too wide to enumerate
// outright (see GreedyCompositionPolicy) -- IntegerBisect reuses the same
// linear-width formula as Continuous, so its sub-box boundaries generally
// land off the integer lattice.
template<typename T>
__device__ __forceinline__ T sample_discrete_from_interval(T lb,
                                                           T ub,
                                                           std::size_t vid,
                                                           std::size_t i,
                                                           std::size_t region,
                                                           std::size_t salt)
{
  std::uint64_t x = sample_hash(vid, i, region, salt);

  T lo = ceil(lb);
  T hi = floor(ub);
  if (hi < lo) {
    return lo;  // sub-box narrower than 1 unit; no integer strictly inside,
                // clamp
  }

  std::size_t count = static_cast<std::size_t>(hi - lo) + 1;
  std::size_t idx =
      static_cast<std::size_t>(x % static_cast<std::uint64_t>(count));
  return lo + static_cast<T>(idx);
}

/**
 * @brief Samples points uniformly from each subdomain
 *
 * `sample_points` is a kernel argument rather than a template parameter: it
 * only ever appeared here, as this loop's bound and the row stride into
 * var_buffers, so making it runtime costs the full unroll of a loop that is
 * not the hot path (the DAG evaluation kernels dominate, and they see only
 * n_elems). Everything else that used it -- n_elems, every buffer size --
 * was already runtime arithmetic.
 *
 * @tparam T numerical precision
 * @tparam CycleSize number of variables being cycled
 * @param parent_domain interval domain being divided
 * @param slot_var_ids  which variable each of the CycleSize slots acts on
 * @param slot_fan_out  fan-out (radix) of each slot
 * @param slot_kind     operation each slot performs
 * @param var_kinds     per-variable kind (Continuous samples get a uniform
 *                      real value; Integer/Binary get a uniform integer)
 * @param var_buffers buffer for sampled points
 * @param n_vars
 * @param n_regions
 * @param sample_points number of points sampled per subdomain
 * @param salt per-launch seed component, so re-visiting a box draws fresh
 * points
 */
template<typename T, std::size_t CycleSize>
__global__ void sample_points_kernel(
    const cu::interval<T>* __restrict parent_domain,
    const std::size_t* __restrict__ slot_var_ids,
    const std::uint32_t* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind,
    const VarKind* __restrict__ var_kinds,
    T* const* __restrict__ var_buffers,  // NUM_VARS x NUM_REGIONS x
                                         // sample_points
    std::size_t n_vars,
    std::size_t n_regions,
    std::size_t slot_count,
    std::size_t sample_points,
    std::size_t salt)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t r = tid; r < n_regions; r += num_threads) {
    auto ctx = partition::make_slot_context<CycleSize>(
        r, slot_var_ids, slot_fan_out, slot_kind, slot_count);
    for (std::size_t vid = 0; vid < n_vars; ++vid) {
      cu::interval<T> v;
      partition::get_slot_bounds(ctx, parent_domain, vid, v);
      bool discrete = var_kinds[vid] != VarKind::Continuous;
      for (std::size_t i = 0; i < sample_points; ++i) {
        T p = discrete
            ? sample_discrete_from_interval(v.lb, v.ub, vid, i, r, salt)
            : sample_from_interval(v.lb, v.ub, vid, i, r, salt);
        var_buffers[vid][r * sample_points + i] = p;
      }
    }
  }
}

// Functor wrappers for Op

// Each tag overloads apply() for interval-interval, interval-scalar, and
// scalar-interval operands. min/max lack a mixed overload in cuinterval, so
// their scalar apply() wraps the scalar as a point interval.
struct AddOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a + b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a + b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a + b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a + b;
  }
};

struct SubOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a - b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a - b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a - b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a - b;
  }
};

struct MulOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a * b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a * b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a * b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a * b;
  }
};

struct DivOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a / b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a / b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a / b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a / b;
  }
};

struct MinOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return min(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return min(a, cu::interval<T>(b));
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return min(cu::interval<T>(a), b);
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return cu::intrinsic::min<T>(a, b);
  }
};

struct MaxOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return max(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return max(a, cu::interval<T>(b));
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return max(cu::interval<T>(a), b);
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return cu::intrinsic::max<T>(a, b);
  }
};

// a^b with both a and b general (interval or scalar), as opposed to
// pown_kernel's integer-exponent path below. cu::pow(interval, interval)
// clips a negative-base domain to [0, +inf) internally, same as cu::sqrt/
// cu::log do for their domains, so no extra domain handling is needed here.
struct PowOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return pow(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return pow(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return pow(cu::interval<T>(a), b);
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    if constexpr (std::is_same_v<T, float>) {
      return ::powf(a, b);
    } else {
      using std::pow;
      return pow(a, b);
    }
  }
};

struct NegOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return -a;
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return -a;
  }
};

struct SqrOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return sqr(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return a * a;
  }
};

struct ExpOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return exp(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::exp(a);
  }
};

struct LogOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return log(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::log(a);
  }
};

struct SqrtOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return sqrt(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::sqrt(a);
  }
};

struct SinOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return sin(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::sin(a);
  }
};

struct CosOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return cos(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::cos(a);
  }
};

struct TanhOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return tanh(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::tanh(a);
  }
};

struct AbsOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return abs(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::fabs(a);
  }
};

// Maps each Op functor tag back to the DAGNode::Op it implements, so
// wire_binary/wire_unary can assert the node they were handed actually
// matches the template parameter the ensure_node switch dispatched to.
template<class OpTag>
struct op_code;

template<>
struct op_code<AddOp>
{
  static constexpr Op value = Op::Add;
};

template<>
struct op_code<SubOp>
{
  static constexpr Op value = Op::Sub;
};

template<>
struct op_code<MulOp>
{
  static constexpr Op value = Op::Mul;
};

template<>
struct op_code<DivOp>
{
  static constexpr Op value = Op::Div;
};

template<>
struct op_code<MinOp>
{
  static constexpr Op value = Op::Min;
};

template<>
struct op_code<MaxOp>
{
  static constexpr Op value = Op::Max;
};

template<>
struct op_code<PowOp>
{
  static constexpr Op value = Op::Pow;
};

template<>
struct op_code<NegOp>
{
  static constexpr Op value = Op::Neg;
};

template<>
struct op_code<SqrOp>
{
  static constexpr Op value = Op::Sqr;
};

template<>
struct op_code<ExpOp>
{
  static constexpr Op value = Op::Exp;
};

template<>
struct op_code<LogOp>
{
  static constexpr Op value = Op::Log;
};

template<>
struct op_code<SqrtOp>
{
  static constexpr Op value = Op::Sqrt;
};

template<>
struct op_code<SinOp>
{
  static constexpr Op value = Op::Sin;
};

template<>
struct op_code<CosOp>
{
  static constexpr Op value = Op::Cos;
};

template<>
struct op_code<TanhOp>
{
  static constexpr Op value = Op::Tanh;
};

template<>
struct op_code<AbsOp>
{
  static constexpr Op value = Op::Abs;
};

template<typename T>
__device__ T pown_scalar(T base, int exponent)
{
  if constexpr (std::is_same_v<T, float>) {
    return ::powf(base, exponent);
  } else {
    using std::pow;
    return pow(base, exponent);
  }
}

// Each of the kernels below applies the templated operation across its input
// intervals, and stores the result in the output T is float or double X is T or
// interval<T>
template<typename X, typename T, class UnaryOp>
__global__ void unary_op_kernel(const X* __restrict__ a,
                                X* __restrict out,
                                std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = UnaryOp::apply(a[i]);
  }
}

template<typename X, typename T, class BinaryOp>
__global__ void binary_op_kernel(const X* __restrict__ a,
                                 const X* __restrict__ b,
                                 X* __restrict out,
                                 std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = BinaryOp::apply(a[i], b[i]);
  }
}

// b is an Op::Const operand's payload, passed by value -- never materialised
// into an n_regions buffer
template<typename X, typename T, class BinaryOp>
__global__ void binary_op_scalar_rhs_kernel(const X* __restrict__ a,
                                            T b,
                                            X* __restrict__ out,
                                            std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = BinaryOp::apply(a[i], b);
  }
}

// a is an Op::Const operand's payload, passed by value -- never materialised
// into an n_regions buffer
template<typename X, typename T, class BinaryOp>
__global__ void binary_op_scalar_lhs_kernel(T a,
                                            const X* __restrict__ b,
                                            X* __restrict__ out,
                                            std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = BinaryOp::apply(a, b[i]);
  }
}

// Both a and b are Op::Const operands' payloads (e.g. two Problem::fixed()
// values combined directly, `p.fixed(2.0) + p.fixed(3.0)`); the result is
// the same broadcast value for every region, but still gets a normal buffer
// so downstream nodes can consume it uniformly.
template<typename X, typename T, class BinaryOp>
__global__ void binary_op_scalar_scalar_kernel(T a,
                                               T b,
                                               X* __restrict__ out,
                                               std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;
  X result = BinaryOp::apply(X(a), X(b));

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = result;
  }
}

// pown_value dispatches to interval pown() or the scalar pown_scalar()
// depending on which V this graph is templated on.
template<typename T>
__device__ cu::interval<T> pown_value(cu::interval<T> a, int exponent)
{
  return pown(a, exponent);
}

template<typename T>
__device__ T pown_value(T a, int exponent)
{
  return pown_scalar(a, exponent);
}

template<typename T, typename V>
__global__ void pown_kernel(const V* __restrict__ a,
                            int exponent,
                            V* __restrict__ out,
                            std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = pown_value(a[i], exponent);
  }
}

template<typename T>
__device__ bool almost_equal(T a, T b, T abs_tol = T(1e-8), T rel_tol = T(1e-6))
{
  T diff = std::abs(a - b);

  if (diff <= abs_tol) {
    return true;
  }

  return diff <= rel_tol * std::max(std::abs(a), std::abs(b));
}

// is_feasible dispatches on V: for intervals this is a sound bound check
// (lb <= rhs, or lb/ub straddling rhs for EQ); for points it's an exact
// (caveat floating point precision) check via almost_equal.
template<typename T>
__device__ bool is_feasible(cu::interval<T> v, Cmp cmp, T rhs)
{
  if (cmp == Cmp::LE) {
    return v.lb <= rhs;
  }
  if (cmp == Cmp::EQ) {
    return v.lb <= rhs && rhs <= v.ub;
  }
  return false;
}

template<typename T>
__device__ bool is_feasible(T v, Cmp cmp, T rhs)
{
  if (cmp == Cmp::LE) {
    return v <= rhs;
  }
  if (cmp == Cmp::EQ) {
    return almost_equal(v, rhs);
  }
  return false;
}

// Writes the shared, absorbing-zero feasible[] flag; concurrent writes agree,
// so no atomics needed. This is strictly an infeasibility check, feasible[i]
// stays 1 if at least part of the bound on the lhs is feasible.
template<typename T, typename V>
__global__ void feasibility_check_kernel(const V* __restrict__ lhs,
                                         Cmp cmp,
                                         T rhs,
                                         unsigned char* __restrict__ feasible,
                                         std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    if (!is_feasible(lhs[i], cmp, rhs)) {
      feasible[i] = 0;
    }
  }
}

// lower_bound_of/upper_bound_of dispatch on V: for intervals these read the
// sound lb/ub; for points both are the identity, so the point graph keeps
// the same node shape as the interval graph despite there being nothing to
// bound.
template<typename T>
__device__ T lower_bound_of(cu::interval<T> v)
{
  return v.lb;
}

template<typename T>
__device__ T lower_bound_of(T v)
{
  return v;
}

template<typename T>
__device__ T upper_bound_of(cu::interval<T> v)
{
  return v.ub;
}

template<typename T>
__device__ T upper_bound_of(T v)
{
  return v;
}

// Extracts obj's lower bound into obj_lb[] for every region, feasible or
// not -- interval soundness holds regardless of constraint feasibility.
// Same format for points and intervals so the graphs are the same
template<typename T, typename V>
__global__ void objective_extract_kernel(const V* __restrict__ obj,
                                         T* __restrict__ obj_lb,
                                         std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    obj_lb[i] = lower_bound_of(obj[i]);
  }
}

// Extracts obj's upper bound into obj_ub[] for the in-graph GUB reduction
// Unmasked by feasibility here -- that happens in mask_infeasible_kernel below,
// once feasible[] is fully written.
// Same format for points and intervals so the graphs are the same.
template<typename T, typename V>
__global__ void objective_extract_ub_kernel(const V* __restrict__ obj,
                                            T* __restrict__ obj_ub,
                                            std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    obj_ub[i] = upper_bound_of(obj[i]);
  }
}

// Maps infeasible regions to +inf so they cannot lower the GUB candidate
// Must run after every constraint's feasibility_check_kernel has written
// feasible[]
template<typename T>
__global__ void mask_infeasible_kernel(
    const T* __restrict__ obj_ub,
    const unsigned char* __restrict__ feasible,
    T* __restrict__ masked_ub,
    std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    masked_ub[i] = feasible[i] ? obj_ub[i] : T(INFINITY);
  }
}

// Dispatches on V the same way lower_bound_of/upper_bound_of do, so the
// no-witness case below writes a NaN of the right shape for either graph.
template<typename T>
__device__ void fill_nan(cu::interval<T>& v)
{
  v.lb = T(NAN);
  v.ub = T(NAN);
}

template<typename T>
__device__ void fill_nan(T& v)
{
  v = T(NAN);
}

// Gathers the witness for the candidate: the per-variable values at the flat
// element index the ArgMin reduction picked out of masked_ub[].
//
// CUB seeds the reduction with numeric_limits<T>::max(), so when every element
// was masked infeasible the winner is a masked +inf (or the untouched seed) and
// no witness exists; out[] is filled with NaN rather than an infeasible point.
template<typename T, typename V>
__global__ void gather_candidate_point_kernel(
    V* const* __restrict__ var_buffers,
    const std::int64_t* __restrict__ argmin_index,
    const T* __restrict__ candidate,
    V* __restrict__ out,
    std::size_t n_vars)
{
  std::size_t vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= n_vars) {
    return;
  }

  if (!(*candidate < ::cuda::std::numeric_limits<T>::max()))
  {  // also catches +inf and NaN
    fill_nan(out[vid]);
    return;
  }
  out[vid] = var_buffers[vid][*argmin_index];
}

// Wires a Problem's shared ExprDAG into the kernel nodes of one CUDA graph.
// Contains per-Problem state, shared across each expression (objective or
// constraint), so a node reachable from more than one is allocated and
// evaluated exactly once. V is the value type of a node's buffer, either
// cu::interval<T> for the interval graph, or T for the sample/exact graphs.
// Below the root node partitioning/sampling/enumerating node, the graphs are
// identical as the kernels and operators are overloaded. Exact (only meaningful
// when V=T) selects the deterministic enumerate_points_kernel instead of
// sample_points_kernel's random sampling -- see ExactGraphReplay.
template<typename T, typename V, std::size_t CycleSize, bool Exact = false>
class GraphBuilder
{
  static_assert(
      std::is_same_v<V, T> || std::is_same_v<V, cu::interval<T>>,
      "V must be T (point/exact graph) or cu::interval<T> (interval graph)");
  static constexpr bool is_point = std::is_same_v<V, T>;

public:
  /**
   * @brief Construct a new Graph Builder object.
   *
   * @param problem Problem representation
   * @param domain_buffer Storage for domain being evaluated
   * @param n_regions Number of regions (the composition's fan-out)
   * @param slot_var_ids Device buffer, CycleSize entries: which variable each
   * slot acts on
   * @param slot_fan_out Device buffer, CycleSize entries: fan-out of each slot
   * @param slot_kind Device buffer, CycleSize entries: operation each slot
   * performs
   * @param var_kinds Device buffer, n_vars entries: per-variable kind. Only
   *                  consumed by the point graph's root node (so samples for
   *                  Integer/Binary variables land on the integer lattice);
   *                  the interval graph ignores it.
   * @param sample_points Points sampled per subdomain. Meaningful only for
   *                  the sampling point graph; the interval and exact graphs
   *                  evaluate one element per region and pass 1.
   */
  GraphBuilder(const Problem<T>& problem,
               cu::interval<T>* domain_buffer,
               std::size_t n_regions,
               const std::size_t* slot_var_ids,
               const std::uint32_t* slot_fan_out,
               const SlotKind* slot_kind,
               const VarKind* var_kinds,
               std::size_t sample_points,
               std::size_t slot_count)
      : problem_(problem)
      , n_regions_(n_regions)
      , sample_points_(sample_points)
      , slot_count_(slot_count)
      , n_elems_(
            is_point
                ? n_regions * sample_points
                : n_regions)  // the number of V-typed slots to operate over
      , block_(256)
      , grid_(static_cast<unsigned int>(detail::ceil_div(n_elems_, 256)))
      , root_grid_(static_cast<unsigned int>(detail::ceil_div(n_regions_, 256)))
  {
    detail::check(cudaGraphCreate(&graph_, 0), "cudaGraphCreate");

    buffers_.resize(problem_.graph.nodes.size(), nullptr);
    producer_nodes_.resize(problem_.graph.nodes.size(), nullptr);

    // Every Op::Var node is materialised eagerly in one shared kernel node
    // that partitions the parent domain and scatters it into each variable's
    // buffer. Op::Const nodes get no buffer; their payload is consumed by
    // value at the use site (see wire_binary).
    std::size_t n_vars = problem_.box_bounds.size();
    std::vector<V*> var_buffer_list(n_vars, nullptr);
    for (const auto& node : problem_.graph.nodes) {
      if (node.op == Op::Var) {
        V* buf = detail::alloc_device<V>(n_elems_);
        var_buffer_list[node.payload.var_index] = buf;
        buffers_[node.id] = buf;
      }
    }

    var_buffers_device_ = detail::alloc_device<V*>(n_vars);
    detail::check(cudaMemcpy(var_buffers_device_,
                             var_buffer_list.data(),
                             n_vars * sizeof(V*),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");

    if constexpr (is_point && Exact) {
      root_node_ =
          detail::add_kernel_node(graph_,
                                  {},
                                  enumerate_points_kernel<T, CycleSize>,
                                  root_grid_,
                                  block_,
                                  domain_buffer,
                                  slot_var_ids,
                                  slot_fan_out,
                                  slot_kind,
                                  var_buffers_device_,
                                  n_vars,
                                  n_regions_,
                                  slot_count_);
    } else if constexpr (is_point) {
      root_node_ = detail::add_kernel_node(graph_,
                                           {},
                                           sample_points_kernel<T, CycleSize>,
                                           root_grid_,
                                           block_,
                                           domain_buffer,
                                           slot_var_ids,
                                           slot_fan_out,
                                           slot_kind,
                                           var_kinds,
                                           var_buffers_device_,
                                           n_vars,
                                           n_regions_,
                                           slot_count_,
                                           sample_points_,
                                           std::size_t {0});
    } else {
      root_node_ =
          detail::add_kernel_node(graph_,
                                  {},
                                  partition_variables_kernel<T, CycleSize>,
                                  root_grid_,
                                  block_,
                                  domain_buffer,
                                  slot_var_ids,
                                  slot_fan_out,
                                  slot_kind,
                                  var_buffers_device_,
                                  n_vars,
                                  n_regions_,
                                  slot_count_);
    }

    for (const auto& node : problem_.graph.nodes) {
      if (node.op == Op::Var) {
        producer_nodes_[node.id] = root_node_;
      }
    }
  }

  // Idempotent: ensures node `id`'s buffer and producing kernel node exist,
  // recursing into `.in` first. Returns the producing graph node (the shared
  // partition node for Op::Var, or the op kernel node otherwise).
  cudaGraphNode_t ensure_node(std::size_t id)
  {
    if (producer_nodes_[id] != nullptr) {
      return producer_nodes_[id];
    }

    const DAGNode<T>& node = problem_.graph.nodes[id];
    switch (node.op) {
      case Op::Const:
        throw std::runtime_error("ensure_node called on an Op::Const node; constants are consumed "
                                 "by value at their use site, never given their own graph node");
      case Op::Var:
        throw std::runtime_error("Op::Var node missing its producer; GraphBuilder constructor "
                                 "invariant broken");
      case Op::Add:
        producer_nodes_[id] = wire_binary<AddOp>(id);
        break;
      case Op::Sub:
        producer_nodes_[id] = wire_binary<SubOp>(id);
        break;
      case Op::Mul:
        producer_nodes_[id] = wire_binary<MulOp>(id);
        break;
      case Op::Div:
        producer_nodes_[id] = wire_binary<DivOp>(id);
        break;
      case Op::Min:
        producer_nodes_[id] = wire_binary<MinOp>(id);
        break;
      case Op::Max:
        producer_nodes_[id] = wire_binary<MaxOp>(id);
        break;
      case Op::Pow:
        producer_nodes_[id] = wire_binary<PowOp>(id);
        break;
      case Op::Neg:
        producer_nodes_[id] = wire_unary<NegOp>(id);
        break;
      case Op::Sqr:
        producer_nodes_[id] = wire_unary<SqrOp>(id);
        break;
      case Op::Exp:
        producer_nodes_[id] = wire_unary<ExpOp>(id);
        break;
      case Op::Log:
        producer_nodes_[id] = wire_unary<LogOp>(id);
        break;
      case Op::Sqrt:
        producer_nodes_[id] = wire_unary<SqrtOp>(id);
        break;
      case Op::Sin:
        producer_nodes_[id] = wire_unary<SinOp>(id);
        break;
      case Op::Cos:
        producer_nodes_[id] = wire_unary<CosOp>(id);
        break;
      case Op::Tanh:
        producer_nodes_[id] = wire_unary<TanhOp>(id);
        break;
      case Op::Abs:
        producer_nodes_[id] = wire_unary<AbsOp>(id);
        break;
      case Op::PowN:
        producer_nodes_[id] = wire_pown(id);
        break;
    }
    return producer_nodes_[id];
  }

  // Walks every node reachable from `root_id` (one function's expression)
  // and wires it into the graph, returning the node producing root_id's
  // buffer.
  cudaGraphNode_t add_expression(std::size_t root_id)
  {
    if (problem_.graph.nodes[root_id].op == Op::Const) {
      throw std::runtime_error("add_expression called on an Op::Const root; not producible via "
                               "the current Expr API (constant() is private, only ever emitted as "
                               "an immediate operand of a binary op)");
    }
    return ensure_node(root_id);
  }

  V* buffer_for(std::size_t id) const { return buffers_[id]; }

  cudaGraph_t graph() const { return graph_; }

  cudaGraphNode_t root_node() const { return root_node_; }

  V** var_buffers_device() const { return var_buffers_device_; }

  dim3 grid() const { return grid_; }

  dim3 root_grid() const { return root_grid_; }

  dim3 block() const { return block_; }

  std::size_t n_elems() const { return n_elems_; }

  // Yields ownership of the per-node buffers to the caller (GraphReplay).
  // GraphBuilder itself never frees anything.
  std::vector<V*> take_node_buffers() { return std::move(buffers_); }

private:
  template<class BinaryOp>
  cudaGraphNode_t wire_binary(std::size_t id)
  {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    if (node.op != op_code<BinaryOp>::value) {
      throw std::runtime_error("wire_binary<BinaryOp> called on a node whose op does not match "
                               "BinaryOp; ensure_node's dispatch switch is out of sync with the "
                               "op_code<> trait");
    }
    if (node.in.size() != 2) {
      throw std::runtime_error(
          "wire_binary called on a node without exactly two operands");
    }
    std::size_t lhs_id = node.in[0];
    std::size_t rhs_id = node.in[1];
    bool lhs_const = problem_.graph.nodes[lhs_id].op == Op::Const;
    bool rhs_const = problem_.graph.nodes[rhs_id].op == Op::Const;

    buffers_[id] = detail::alloc_device<V>(n_elems_);

    if (!lhs_const && !rhs_const) {
      cudaGraphNode_t lhs_node = ensure_node(lhs_id);
      cudaGraphNode_t rhs_node = ensure_node(rhs_id);
      // Both operands can share the same producer (e.g. `x + y` where x and y
      // are both Op::Var, produced by the one shared partition node), and
      // cudaGraphAddKernelNode rejects a duplicated entry in the dependency
      // list, so dedup before adding.
      std::vector<cudaGraphNode_t> deps = (lhs_node == rhs_node)
          ? std::vector<cudaGraphNode_t> {lhs_node}
          : std::vector<cudaGraphNode_t> {lhs_node, rhs_node};
      return detail::add_kernel_node(graph_,
                                     deps,
                                     binary_op_kernel<V, T, BinaryOp>,
                                     grid_,
                                     block_,
                                     buffers_[lhs_id],
                                     buffers_[rhs_id],
                                     buffers_[id],
                                     n_elems_);
    }
    if (!lhs_const && rhs_const) {
      cudaGraphNode_t lhs_node = ensure_node(lhs_id);
      T rhs_val = problem_.graph.nodes[rhs_id].payload.constant;
      return detail::add_kernel_node(
          graph_,
          {lhs_node},
          binary_op_scalar_rhs_kernel<V, T, BinaryOp>,
          grid_,
          block_,
          buffers_[lhs_id],
          rhs_val,
          buffers_[id],
          n_elems_);
    }
    if (lhs_const && !rhs_const) {
      T lhs_val = problem_.graph.nodes[lhs_id].payload.constant;
      cudaGraphNode_t rhs_node = ensure_node(rhs_id);
      return detail::add_kernel_node(
          graph_,
          {rhs_node},
          binary_op_scalar_lhs_kernel<V, T, BinaryOp>,
          grid_,
          block_,
          lhs_val,
          buffers_[rhs_id],
          buffers_[id],
          n_elems_);
    }
    // Both operands are Op::Const (e.g. two Problem::fixed() values combined
    // directly): the result doesn't depend on the domain or any other node,
    // so it's materialised with no dependencies, same as root_node_.
    T lhs_val = problem_.graph.nodes[lhs_id].payload.constant;
    T rhs_val = problem_.graph.nodes[rhs_id].payload.constant;
    return detail::add_kernel_node(
        graph_,
        {},
        binary_op_scalar_scalar_kernel<V, T, BinaryOp>,
        grid_,
        block_,
        lhs_val,
        rhs_val,
        buffers_[id],
        n_elems_);
  }

  template<class UnaryOp>
  cudaGraphNode_t wire_unary(std::size_t id)
  {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    if (node.op != op_code<UnaryOp>::value) {
      throw std::runtime_error("wire_unary<UnaryOp> called on a node whose op does not match "
                               "UnaryOp; ensure_node's dispatch switch is out of sync with the "
                               "op_code<> trait");
    }
    if (node.in.size() != 1) {
      throw std::runtime_error(
          "wire_unary called on a node without exactly one operand");
    }
    std::size_t operand_id = node.in[0];
    if (problem_.graph.nodes[operand_id].op == Op::Const) {
      throw std::runtime_error("unary op applied directly to an Op::Const; unreachable via the "
                               "current Expr API");
    }
    cudaGraphNode_t operand_node = ensure_node(operand_id);
    buffers_[id] = detail::alloc_device<V>(n_elems_);
    return detail::add_kernel_node(graph_,
                                   {operand_node},
                                   unary_op_kernel<V, T, UnaryOp>,
                                   grid_,
                                   block_,
                                   buffers_[operand_id],
                                   buffers_[id],
                                   n_elems_);
  }

  cudaGraphNode_t wire_pown(std::size_t id)
  {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    if (node.op != Op::PowN) {
      throw std::runtime_error(
          "wire_pown called on a node whose op is not Op::PowN");
    }
    if (node.in.size() != 1) {
      throw std::runtime_error(
          "wire_pown called on a node without exactly one operand");
    }
    std::size_t operand_id = node.in[0];
    if (problem_.graph.nodes[operand_id].op == Op::Const) {
      throw std::runtime_error("Op::PowN applied directly to an Op::Const; unreachable via the "
                               "current Expr API");
    }
    cudaGraphNode_t operand_node = ensure_node(operand_id);
    buffers_[id] = detail::alloc_device<V>(n_elems_);
    return detail::add_kernel_node(graph_,
                                   {operand_node},
                                   pown_kernel<T, V>,
                                   grid_,
                                   block_,
                                   buffers_[operand_id],
                                   node.payload.int_exp,
                                   buffers_[id],
                                   n_elems_);
  }

  const Problem<T>& problem_;
  std::size_t n_regions_;
  // A member, not a ctor local: add_kernel_node takes the address of what it
  // is given, so the value must outlive the call that bakes it into the
  // graph node.
  std::size_t sample_points_;
  std::size_t slot_count_;
  std::size_t n_elems_;
  dim3 block_;
  dim3 grid_;
  dim3 root_grid_;
  cudaGraph_t graph_ {};
  cudaGraphNode_t root_node_ {};
  V** var_buffers_device_ = nullptr;
  std::vector<V*> buffers_;  // indexed by node id, null until allocated
  std::vector<cudaGraphNode_t>
      producer_nodes_;  // indexed by node id, null until added
};

// ---------------------------------------------------------------------------
// Device-memory sizing
//
// These are free functions rather than GraphReplay members because the
// quantity that actually decides whether a configuration runs is not any one
// graph's footprint but the footprint of *every* graph a solve holds at once.
// GraphDriver caches, per Composition it encounters, a point graph, an
// interval graph, and -- when the Composition is fully enumerable -- an exact
// graph, and those are live simultaneously.
//
// Sizing against a single graph is what made the old --max-cycle-size
// suggestion unreachable. The exact graph evaluates one element per region,
// so when it overflowed, its budget scan recommended a cap at which the point
// graph -- sample_points elements per region -- promptly overflowed in turn,
// and the caller got a second out-of-memory failure for having followed the
// advice in the first.
// ---------------------------------------------------------------------------

// buffer_node_count/element_bytes/composition_footprint_bytes/
// auto_max_cycle_size used to be defined here. They moved to
// search_sizing.hpp (included above) so a host-only translation unit can
// reuse them without this file's __global__ kernels and <cub/cub.cuh> --
// see that header's top comment.

// Owns the entire graph replay for a problem and value-type V (either T or
// cu::interval<T>) This includes:
// - The root node (partition for intervals, sample for points)
// - Every op-kernel node
// - A feasibility check kernel per constraint
// - An objective extraction kernel
// Main usage is `build`, `set_domain` and `launch`.
// Each instance owns device memory, a cudaGraph_t and a cudaGraphExec_t
// Exact (only meaningful when V=T) selects the deterministic
// enumerate_points_kernel as the root node instead of sample_points_kernel's
// random sampling -- see ExactGraphReplay/GraphBuilder.
template<typename T, typename V, std::size_t CycleSize, bool Exact = false>
class GraphReplay
{
  static constexpr bool is_point = std::is_same_v<V, T>;

public:
  /// @copydoc cuminlp::dag::buffer_node_count
  /// Kept as a member for callers that already have a replay type in hand;
  /// the implementation is shared with the free sizing functions above.
  static std::size_t count_buffer_nodes(const Problem<T>& problem)
  {
    return buffer_node_count(problem);
  }

  /**
   * @brief Device bytes build() would allocate for a composition this wide.
   *
   * Every buffer this class owns is sized off `n_elems` (= n_regions, times
   * sample_points for the point graph), and there is one such buffer per
   * reachable non-Const DAG node. So the total is linear in n_elems with a
   * coefficient the Problem fixes, and can be computed before allocating
   * anything.
   *
   * Saturates at SIZE_MAX, for the same reason composition_fan_out does:
   * with partition_num now a runtime value, `n_regions` can arrive already
   * saturated, and a wrapped byte count would read as comfortably in budget.
   */
  static std::size_t estimate_bytes(const Problem<T>& problem,
                                    std::size_t n_regions,
                                    std::size_t sample_points = 1)
  {
    return bytes_for(n_regions, sample_points, count_buffer_nodes(problem));
  }

  // Bytes per element: one V per buffer-bearing node, plus feasible[] (1
  // byte) and obj_lb/obj_ub/masked_ub (T each).
  static std::size_t bytes_per_element(std::size_t n_buffers)
  {
    return detail::saturating_mul(n_buffers, sizeof(V)) + 1 + 3 * sizeof(T);
  }

  static std::size_t bytes_for(std::size_t n_regions,
                               std::size_t sample_points,
                               std::size_t n_buffers)
  {
    std::size_t const n_elems = is_point
        ? detail::saturating_mul(n_regions, sample_points)
        : n_regions;
    return detail::saturating_mul(n_elems, bytes_per_element(n_buffers));
  }

  /// Which of the three graph shapes this instantiation is, for diagnostics.
  static const char* graph_kind()
  {
    if constexpr (is_point && Exact) {
      return "exact";
    } else if constexpr (is_point) {
      return "point";
    } else {
      return "interval";
    }
  }

  /**
   * @brief Explain an over-budget build: where the size came from, and which
   *        knob to turn.
   *
   * The region count is a *product* over slots, so an out-of-budget request
   * is usually out by orders of magnitude and the raw byte figure alone tells
   * a caller nothing actionable. This shows the multiplication that produced
   * it, then does the arithmetic the caller would otherwise have to: the
   * widest slot cap that would actually fit.
   *
   * That last part is the useful half, and it is costed against the whole
   * *problem*, not against prefixes of the composition that happened to fail.
   * Truncating this composition is what made the suggestion unfollowable a
   * second time over: on a problem with both binaries and continuous
   * variables the failing composition's cheap prefix (binaries at fan-out 2)
   * is not what a cap of that size buys further down the tree, where the
   * resolved binaries have handed their slots to continuous variables at
   * `partition_num` each. `dag::auto_max_cycle_size` charges the widest
   * composition the search can still reach at each cap -- see
   * search_sizing.hpp.
   *
   * It also charges composition_footprint_bytes -- every graph a solve holds
   * for a composition at once -- not this graph alone. Scanning this graph
   * alone is what made the suggestion unfollowable the *first* time: an exact
   * graph overflowing at one element per region would recommend a cap at
   * which the point graph, at `solve_sample_points` elements per region,
   * overflowed in turn.
   *
   * Hence the two sample counts. `sample_points` is what *this* graph draws,
   * and describes the size that failed; `solve_sample_points` is the
   * solve-wide setting, and is what the recommendation has to be costed
   * against. They differ exactly when this is not the point graph.
   */
  static std::string out_of_memory_report(
      const Problem<T>& problem,
      const Composition<CycleSize>& composition,
      const FanOutSpec& fan_out,
      std::size_t sample_points,
      std::size_t n_buffers,
      std::size_t needed,
      std::size_t budget,
      std::size_t solve_sample_points)
  {
    std::size_t const n_regions = composition_fan_out(composition, fan_out);

    std::string msg = std::string(graph_kind()) + " graph needs "
        + detail::format_bytes(needed) + " of device memory, but only "
        + detail::format_bytes(budget) + " is available.\n";

    // Per-kind slot tally: "10 x IntegerEnumerate (fan-out 7 each)".
    msg += "  composition: " + std::to_string(composition.count)
        + " live slot(s) of " + std::to_string(CycleSize) + " compiled";
    for (int k = 0; k < 5; ++k) {
      auto const kind = static_cast<SlotKind>(k);
      if (kind == SlotKind::Padding) {
        continue;
      }
      std::size_t slots = 0;
      for (std::size_t j = 0; j < composition.count; ++j) {
        if (composition[j] == kind) {
          ++slots;
        }
      }
      if (slots > 0) {
        msg += "\n    " + std::to_string(slots) + " x " + slot_kind_name(kind)
            + " (fan-out " + std::to_string(slot_fan_out(kind, fan_out))
            + " each)";
      }
    }

    msg += "\n  -> " + detail::format_count(n_regions) + " regions";
    if (is_point && sample_points > 1) {
      msg += " x " + std::to_string(sample_points) + " sample points";
    }
    msg += "\n  x " + detail::format_bytes(bytes_per_element(n_buffers))
        + " per element (" + std::to_string(n_buffers)
        + " DAG-node buffers of " + std::to_string(sizeof(V))
        + " B, plus " + std::to_string(1 + 3 * sizeof(T))
        + " B of per-element bookkeeping)"
        + "\n  = " + detail::format_bytes(needed) + '\n';

    // Report the widest cap that fits -- costed against every graph the solve
    // would then hold, for the widest composition it could still reach at
    // that cap, so that following the suggestion cannot fail a second time.
    std::size_t n_binary = 0;
    std::size_t n_integer = 0;
    std::size_t n_continuous = 0;
    for (VarKind kind : problem.var_kinds) {
      if (kind == VarKind::Binary) {
        ++n_binary;
      } else if (kind == VarKind::Integer) {
        ++n_integer;
      } else {
        ++n_continuous;
      }
    }

    std::size_t const best_slots = auto_max_cycle_size<T>(n_binary,
                                                          n_integer,
                                                          n_continuous,
                                                          n_buffers,
                                                          fan_out,
                                                          solve_sample_points,
                                                          budget,
                                                          CycleSize);

    if (best_slots > 0) {
      std::size_t const best_bytes =
          worst_composition_footprint<T>(best_slots,
                                         n_binary,
                                         n_integer,
                                         n_continuous,
                                         n_buffers,
                                         fan_out,
                                         solve_sample_points);
      msg += "\n  Acting on " + std::to_string(best_slots)
          + " variable(s) at a time instead of "
          + std::to_string(composition.count)
          + " keeps every composition the search can reach -- point, interval"
            " and exact graphs together -- within "
          + detail::format_bytes(best_bytes)
          + ": try --max-cycle-size=" + std::to_string(best_slots) + '\n';
    } else {
      msg +=
          "\n  Even a single slot does not fit, so the per-element cost is "
          "the problem rather than the slot count: this Problem needs "
          + detail::format_bytes(bytes_per_element(n_buffers))
          + " per region for its " + std::to_string(n_buffers) + " DAG nodes\n";
    }
    msg +=
        "  Lowering --partition-num / --enumerate-cap shrinks each slot's "
        "fan-out, which reduces the product too.";
    return msg;
  }

  // `composition` fixes this replay's fan-out/shape for its whole lifetime;
  // only which variables fill its slots (set_domain's `var_ids`) varies
  // between launches.
  //
  // `budget_bytes` caps what this build may allocate on the device; 0 means
  // "ask the driver for what's currently free". A composition whose fan-out
  // exceeds it raises ResourceExhausted *before* allocating anything, rather
  // than dying partway through with a bare cudaErrorMemoryAllocation. This
  // guard did not exist while partition_num was a template parameter: the
  // handful of compile-time shapes were hand-tuned to fit, whereas a runtime
  // `--partition-num 10` with CycleSize 20 now asks for 10^20 regions from a
  // command line.
  //
  // `sample_points` is ignored by the interval and exact graphs, which
  // evaluate exactly one element per region; only the sampling point graph
  // reads it.
  static GraphReplay build(const Problem<T>& problem,
                           const Composition<CycleSize>& composition,
                           const FanOutSpec& fan_out,
                           std::size_t budget_bytes = 0,
                           std::size_t sample_points = 1)
  {
    if (sample_points < 1) {
      throw cuminlp::InvalidConfiguration(
          "sample_points must be at least 1; a point graph that samples "
          "nothing can never produce an incumbent");
    }

    GraphReplay replay;
    replay.composition_ = composition;
    replay.n_regions_ = composition_fan_out(composition, fan_out);
    replay.n_vars_ = problem.box_bounds.size();
    replay.sample_points_ = is_point && !Exact ? sample_points : 1;
    replay.slot_count_ = composition.count;

    std::size_t const n_buffers = count_buffer_nodes(problem);
    std::size_t const needed = bytes_for(
        replay.n_regions_, replay.sample_points_, n_buffers);
    if (budget_bytes == 0) {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      detail::check(cudaMemGetInfo(&free_bytes, &total_bytes),
                    "cudaMemGetInfo");
      budget_bytes = free_bytes;
    }
    if (needed > budget_bytes) {
      throw cuminlp::ResourceExhausted(out_of_memory_report(
          problem, composition, fan_out, replay.sample_points_, n_buffers,
          needed, budget_bytes, sample_points));
    }

    replay.domain_buffer_ =
        detail::alloc_device<cu::interval<T>>(replay.n_vars_);
    replay.slot_var_ids_device_ = detail::alloc_device<std::size_t>(CycleSize);
    replay.slot_fan_out_device_ = detail::alloc_device<std::uint32_t>(CycleSize);
    replay.slot_kind_device_ = detail::alloc_device<SlotKind>(CycleSize);
    replay.var_kinds_device_ = detail::alloc_device<VarKind>(replay.n_vars_);

    // fan_out/kind are fixed by `composition`, and var_kinds by the problem,
    // so they're all uploaded once here rather than on every set_domain()
    // call like var_ids.
    std::array<std::uint32_t, CycleSize> fan_out_host {};
    for (std::size_t j = 0; j < CycleSize; ++j) {
      fan_out_host[j] =
          static_cast<std::uint32_t>(slot_fan_out(composition[j], fan_out));
    }
    detail::check(cudaMemcpy(replay.slot_fan_out_device_,
                             fan_out_host.data(),
                             CycleSize * sizeof(std::uint32_t),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(replay.slot_kind_device_,
                             composition.data(),
                             CycleSize * sizeof(SlotKind),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(replay.var_kinds_device_,
                             problem.var_kinds.data(),
                             replay.n_vars_ * sizeof(VarKind),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");

    GraphBuilder<T, V, CycleSize, Exact> builder(problem,
                                                 replay.domain_buffer_,
                                                 replay.n_regions_,
                                                 replay.slot_var_ids_device_,
                                                 replay.slot_fan_out_device_,
                                                 replay.slot_kind_device_,
                                                 replay.var_kinds_device_,
                                                 replay.sample_points_,
                                                 replay.slot_count_);

    // n_elems_ is n_regions_ for the interval graph, n_regions_ * sample_points
    // for the point graph -- every per-node buffer (feasible[], obj_lb[]
    // included) is sized off it, not n_regions_ directly.
    replay.n_elems_ = builder.n_elems();
    replay.feasible_buffer_ =
        detail::alloc_device<unsigned char>(replay.n_elems_);
    replay.obj_lb_buffer_ = detail::alloc_device<T>(replay.n_elems_);
    replay.obj_ub_buffer_ = detail::alloc_device<T>(replay.n_elems_);
    replay.masked_ub_buffer_ = detail::alloc_device<T>(replay.n_elems_);
    replay.candidate_obj_buffer_ = detail::alloc_device<T>(1);
    replay.candidate_index_buffer_ = detail::alloc_device<std::int64_t>(1);
    replay.candidate_point_buffer_ = detail::alloc_device<V>(replay.n_vars_);
    replay.feasible_host_.resize(replay.n_elems_);
    replay.obj_lb_host_.resize(replay.n_elems_);
    replay.candidate_point_host_.resize(replay.n_vars_);

    // Root memset: feasible[] = 1 each replay, race-free
    // fan-out target for every feasibility_check_kernel node.
    cudaMemsetParams memset_params {};
    memset_params.dst = replay.feasible_buffer_;
    memset_params.pitch = 0;
    memset_params.value = 1;
    memset_params.elementSize = 1;
    memset_params.width = replay.n_elems_;
    memset_params.height = 1;
    cudaGraphNode_t feasible_memset_node;
    detail::check(
        cudaGraphAddMemsetNode(
            &feasible_memset_node, builder.graph(), nullptr, 0, &memset_params),
        "cudaGraphAddMemsetNode");

    cudaGraphNode_t obj_producer =
        builder.add_expression(problem.objective_root);

    // Collected so the mask node below can fan in off every constraint's
    // write to feasible[] and ensure serialisation
    std::vector<cudaGraphNode_t> feasibility_nodes;
    feasibility_nodes.reserve(problem.constraints.size());

    for (const auto& constraint : problem.constraints) {
      cudaGraphNode_t constraint_producer =
          builder.add_expression(constraint.root_id);
      cudaGraphNode_t feasibility_node =
          detail::add_kernel_node(builder.graph(),
                                  {constraint_producer, feasible_memset_node},
                                  feasibility_check_kernel<T, V>,
                                  builder.grid(),
                                  builder.block(),
                                  builder.buffer_for(constraint.root_id),
                                  constraint.cmp,
                                  constraint.rhs,
                                  replay.feasible_buffer_,
                                  replay.n_elems_);
      feasibility_nodes.push_back(feasibility_node);
    }

    detail::add_kernel_node(builder.graph(),
                            {obj_producer},
                            objective_extract_kernel<T, V>,
                            builder.grid(),
                            builder.block(),
                            builder.buffer_for(problem.objective_root),
                            replay.obj_lb_buffer_,
                            replay.n_elems_);

    cudaGraphNode_t obj_ub_node =
        detail::add_kernel_node(builder.graph(),
                                {obj_producer},
                                objective_extract_ub_kernel<T, V>,
                                builder.grid(),
                                builder.block(),
                                builder.buffer_for(problem.objective_root),
                                replay.obj_ub_buffer_,
                                replay.n_elems_);

    std::vector<cudaGraphNode_t> mask_deps = feasibility_nodes;
    mask_deps.push_back(feasible_memset_node);
    mask_deps.push_back(obj_ub_node);
    cudaGraphNode_t mask_node =
        detail::add_kernel_node(builder.graph(),
                                mask_deps,
                                mask_infeasible_kernel<T>,
                                builder.grid(),
                                builder.block(),
                                replay.obj_ub_buffer_,
                                replay.feasible_buffer_,
                                replay.masked_ub_buffer_,
                                replay.n_elems_);

    // CUB's DeviceReduce dispatches its own internal kernels rather than
    // exposing one nameable __global__ function, so it can't go through
    // detail::add_kernel_node like everything else here. Instead: capture one
    // call to it on a scratch stream, then fold the captured subgraph in as a
    // single child-graph node. This is a one-time build-time cost, not a
    // per-replay one.
    std::size_t temp_storage_bytes = 0;
    detail::check(
        cub::DeviceReduce::ArgMin(nullptr,
                                  temp_storage_bytes,
                                  replay.masked_ub_buffer_,
                                  replay.candidate_obj_buffer_,
                                  replay.candidate_index_buffer_,
                                  static_cast<std::int64_t>(replay.n_elems_)),
        "cub::DeviceReduce::ArgMin (size query)");
    replay.cub_temp_storage_ =
        detail::alloc_device<unsigned char>(temp_storage_bytes);

    cudaStream_t capture_stream;
    detail::check(cudaStreamCreate(&capture_stream), "cudaStreamCreate");
    detail::check(cudaStreamBeginCapture(capture_stream,
                                         cudaStreamCaptureModeThreadLocal),
                  "cudaStreamBeginCapture");
    detail::check(
        cub::DeviceReduce::ArgMin(replay.cub_temp_storage_,
                                  temp_storage_bytes,
                                  replay.masked_ub_buffer_,
                                  replay.candidate_obj_buffer_,
                                  replay.candidate_index_buffer_,
                                  static_cast<std::int64_t>(replay.n_elems_),
                                  capture_stream),
        "cub::DeviceReduce::ArgMin (capture)");
    cudaGraph_t captured_graph;
    detail::check(cudaStreamEndCapture(capture_stream, &captured_graph),
                  "cudaStreamEndCapture");
    cudaGraphNode_t reduce_node;
    detail::check(
        cudaGraphAddChildGraphNode(
            &reduce_node, builder.graph(), &mask_node, 1, captured_graph),
        "cudaGraphAddChildGraphNode");
    detail::check(cudaGraphDestroy(captured_graph), "cudaGraphDestroy");
    detail::check(cudaStreamDestroy(capture_stream), "cudaStreamDestroy");

    // One thread per variable, reading the index the reduction just wrote, so
    // it must fan in off reduce_node. The var buffers it gathers from are
    // written by root_node_, which reduce_node already transitively depends on.
    // This is the graph's terminal node; nothing downstream depends on it.
    detail::add_kernel_node(
        builder.graph(),
        {reduce_node},
        gather_candidate_point_kernel<T, V>,
        dim3(static_cast<unsigned int>(detail::ceil_div(replay.n_vars_, 256))),
        builder.block(),
        builder.var_buffers_device(),
        static_cast<const std::int64_t*>(replay.candidate_index_buffer_),
        static_cast<const T*>(replay.candidate_obj_buffer_),
        replay.candidate_point_buffer_,
        replay.n_vars_);

    replay.graph_ = builder.graph();
    detail::check(cudaGraphInstantiate(&replay.exec_, replay.graph_, 0),
                  "cudaGraphInstantiate");

    replay.root_node_ = builder.root_node();
    replay.var_buffers_device_ = builder.var_buffers_device();
    replay.root_grid_ = builder.root_grid();
    replay.block_ = builder.block();
    replay.node_buffers_ = builder.take_node_buffers();

    return replay;
  }

  GraphReplay(const GraphReplay&) = delete;
  GraphReplay& operator=(const GraphReplay&) = delete;

  GraphReplay(GraphReplay&& other) noexcept { *this = std::move(other); }

  GraphReplay& operator=(GraphReplay&& other) noexcept
  {
    if (this == &other) {
      return *this;
    }
    free_resources();
    graph_ = other.graph_;
    exec_ = other.exec_;
    domain_buffer_ = other.domain_buffer_;
    feasible_buffer_ = other.feasible_buffer_;
    obj_lb_buffer_ = other.obj_lb_buffer_;
    obj_ub_buffer_ = other.obj_ub_buffer_;
    masked_ub_buffer_ = other.masked_ub_buffer_;
    candidate_obj_buffer_ = other.candidate_obj_buffer_;
    candidate_index_buffer_ = other.candidate_index_buffer_;
    candidate_point_buffer_ = other.candidate_point_buffer_;
    cub_temp_storage_ = other.cub_temp_storage_;
    var_buffers_device_ = other.var_buffers_device_;
    slot_var_ids_device_ = other.slot_var_ids_device_;
    slot_fan_out_device_ = other.slot_fan_out_device_;
    slot_kind_device_ = other.slot_kind_device_;
    var_kinds_device_ = other.var_kinds_device_;
    root_node_ = other.root_node_;
    composition_ = other.composition_;
    n_regions_ = other.n_regions_;
    n_elems_ = other.n_elems_;

    sample_points_ = other.sample_points_;
    slot_count_ = other.slot_count_;
    n_vars_ = other.n_vars_;
    root_grid_ = other.root_grid_;
    block_ = other.block_;
    node_buffers_ = std::move(other.node_buffers_);
    feasible_host_ = std::move(other.feasible_host_);
    obj_lb_host_ = std::move(other.obj_lb_host_);
    candidate_point_host_ = std::move(other.candidate_point_host_);
    candidate_host_ = other.candidate_host_;
    candidate_index_host_ = other.candidate_index_host_;
    other.graph_ = nullptr;
    other.exec_ = nullptr;
    other.domain_buffer_ = nullptr;
    other.feasible_buffer_ = nullptr;
    other.obj_lb_buffer_ = nullptr;
    other.obj_ub_buffer_ = nullptr;
    other.masked_ub_buffer_ = nullptr;
    other.candidate_obj_buffer_ = nullptr;
    other.candidate_index_buffer_ = nullptr;
    other.candidate_point_buffer_ = nullptr;
    other.cub_temp_storage_ = nullptr;
    other.var_buffers_device_ = nullptr;
    other.slot_var_ids_device_ = nullptr;
    other.slot_fan_out_device_ = nullptr;
    other.slot_kind_device_ = nullptr;
    other.var_kinds_device_ = nullptr;
    other.root_node_ = nullptr;
    return *this;
  }

  ~GraphReplay() { free_resources(); }

  // Driver calls this before each launch to update the parent domain and
  // which variable fills each slot. Both update via plain memcpy into fixed
  // buffer addresses (slot_fan_out_/slot_kind_ don't need re-uploading --
  // they're fixed by this replay's composition); the kernel args are
  // re-baked via cudaGraphExecKernelNodeSetParams regardless, since the API
  // requires the whole arg list on every update, not just what changed.
  // `salt` seeds the point graph's sampler (see sample_from_interval); the
  // interval graph's root kernel takes no such argument and ignores it.
  void set_domain(std::span<const cu::interval<T>> domain,
                  const std::array<std::size_t, CycleSize>& var_ids,
                  std::size_t salt = 0)
  {
    if (exec_ == nullptr) {
      throw cuminlp::error("set_domain() called on a moved-from GraphReplay");
    }
    if (domain.size() != n_vars_) {
      throw cuminlp::ShapeMismatch(
          "set_domain: domain size does not match the problem's variable "
          "count");
    }
    detail::check(cudaMemcpy(domain_buffer_,
                             domain.data(),
                             n_vars_ * sizeof(cu::interval<T>),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(slot_var_ids_device_,
                             var_ids.data(),
                             CycleSize * sizeof(std::size_t),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");

    cudaKernelNodeParams params {};
    params.gridDim = root_grid_;
    params.blockDim = block_;
    params.sharedMemBytes = 0;
    params.extra = nullptr;

    // The root kernels differ in arity, so the argument arrays do too.
    if constexpr (is_point && Exact) {
      void* kernel_args[] = {
          &domain_buffer_,
          &slot_var_ids_device_,
          &slot_fan_out_device_,
          &slot_kind_device_,
          &var_buffers_device_,
          &n_vars_,
          &n_regions_,
          &slot_count_,
      };
      params.func =
          reinterpret_cast<void*>(enumerate_points_kernel<T, CycleSize>);
      params.kernelParams = kernel_args;
      detail::check(
          cudaGraphExecKernelNodeSetParams(exec_, root_node_, &params),
          "cudaGraphExecKernelNodeSetParams");
    } else if constexpr (is_point) {
      void* kernel_args[] = {
          &domain_buffer_,
          &slot_var_ids_device_,
          &slot_fan_out_device_,
          &slot_kind_device_,
          &var_kinds_device_,
          &var_buffers_device_,
          &n_vars_,
          &n_regions_,
          &slot_count_,
          &sample_points_,
          &salt,
      };
      params.func =
          reinterpret_cast<void*>(sample_points_kernel<T, CycleSize>);
      params.kernelParams = kernel_args;
      detail::check(
          cudaGraphExecKernelNodeSetParams(exec_, root_node_, &params),
          "cudaGraphExecKernelNodeSetParams");
    } else {
      void* kernel_args[] = {
          &domain_buffer_,
          &slot_var_ids_device_,
          &slot_fan_out_device_,
          &slot_kind_device_,
          &var_buffers_device_,
          &n_vars_,
          &n_regions_,
          &slot_count_,
      };
      params.func =
          reinterpret_cast<void*>(partition_variables_kernel<T, CycleSize>);
      params.kernelParams = kernel_args;
      detail::check(
          cudaGraphExecKernelNodeSetParams(exec_, root_node_, &params),
          "cudaGraphExecKernelNodeSetParams");
    }
  }

  // Launches the graph and synchronises; feasible[]/obj_lb[]/candidate()/
  // candidate_point() D2H copy is a manual cudaMemcpy for now. candidate() is
  // this launch's own feasibility-masked min objective upper bound, and
  // candidate_point() the variable values that attained it.
  void launch(cudaStream_t stream)
  {
    if (exec_ == nullptr) {
      throw cuminlp::error("launch() called on a moved-from GraphReplay");
    }
    detail::check(cudaGraphLaunch(exec_, stream), "cudaGraphLaunch");
    detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    detail::check(cudaMemcpy(feasible_host_.data(),
                             feasible_buffer_,
                             n_elems_ * sizeof(unsigned char),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(obj_lb_host_.data(),
                             obj_lb_buffer_,
                             n_elems_ * sizeof(T),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(&candidate_host_,
                             candidate_obj_buffer_,
                             sizeof(T),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(&candidate_index_host_,
                             candidate_index_buffer_,
                             sizeof(std::int64_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(candidate_point_host_.data(),
                             candidate_point_buffer_,
                             n_vars_ * sizeof(V),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
  }

  std::span<const unsigned char> feasible() const { return feasible_host_; }

  std::span<const T> obj_lb() const { return obj_lb_host_; }

  T candidate() const { return candidate_host_; }

  // The argmin witness for candidate(), indexed by variable: for the point
  // graph the exact sampled values, for the interval graph the argmin sub-box.
  // All-NaN when this launch found no feasible element (candidate() is then
  // CUB's numeric_limits<T>::max() seed) -- check has_candidate() first.
  std::span<const V> candidate_point() const { return candidate_point_host_; }

  // Flat element index the reduction picked; r * sample_points + i for the
  // point graph, the region index for the interval graph.
  std::int64_t candidate_index() const { return candidate_index_host_; }

  bool has_candidate() const
  {
    return candidate_host_ < std::numeric_limits<T>::max();
  }

  std::size_t n_regions() const { return n_regions_; }

  std::size_t n_elems() const { return n_elems_; }

  std::size_t n_vars() const { return n_vars_; }

  const Composition<CycleSize>& composition() const { return composition_; }

private:
  GraphReplay() = default;

  void free_resources()
  {
    if (exec_) {
      cudaGraphExecDestroy(exec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
    if (domain_buffer_) {
      cudaFree(domain_buffer_);
    }
    if (feasible_buffer_) {
      cudaFree(feasible_buffer_);
    }
    if (obj_lb_buffer_) {
      cudaFree(obj_lb_buffer_);
    }
    if (obj_ub_buffer_) {
      cudaFree(obj_ub_buffer_);
    }
    if (masked_ub_buffer_) {
      cudaFree(masked_ub_buffer_);
    }
    if (candidate_obj_buffer_) {
      cudaFree(candidate_obj_buffer_);
    }
    if (candidate_index_buffer_) {
      cudaFree(candidate_index_buffer_);
    }
    if (candidate_point_buffer_) {
      cudaFree(candidate_point_buffer_);
    }
    if (cub_temp_storage_) {
      cudaFree(cub_temp_storage_);
    }
    if (var_buffers_device_) {
      cudaFree(var_buffers_device_);
    }
    if (slot_var_ids_device_) {
      cudaFree(slot_var_ids_device_);
    }
    if (slot_fan_out_device_) {
      cudaFree(slot_fan_out_device_);
    }
    if (slot_kind_device_) {
      cudaFree(slot_kind_device_);
    }
    if (var_kinds_device_) {
      cudaFree(var_kinds_device_);
    }
    for (auto* buf : node_buffers_) {
      if (buf) {
        cudaFree(buf);
      }
    }
  }

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  cudaGraphNode_t root_node_ = nullptr;
  cu::interval<T>* domain_buffer_ = nullptr;  // device, written by set_domain()
  unsigned char* feasible_buffer_ = nullptr;  // device
  T* obj_lb_buffer_ = nullptr;  // device, D2H-copied after each launch()
  T* obj_ub_buffer_ = nullptr;  // device, feeds mask_infeasible_kernel
  T* masked_ub_buffer_ = nullptr;  // device, feeds the CUB reduction
  T* candidate_obj_buffer_ = nullptr;  // device, CUB reduction output (scalar)
  std::int64_t* candidate_index_buffer_ =
      nullptr;  // device, CUB reduction output (flat index)
  V* candidate_point_buffer_ =
      nullptr;  // device, n_vars_, gathered at the argmin index
  unsigned char* cub_temp_storage_ =
      nullptr;  // device, CUB scratch, sized once at build
  V** var_buffers_device_ = nullptr;
  std::size_t* slot_var_ids_device_ =
      nullptr;  // device, CycleSize entries, written by set_domain()
  std::uint32_t* slot_fan_out_device_ =
      nullptr;  // device, CycleSize entries, fixed by composition_
  SlotKind* slot_kind_device_ =
      nullptr;  // device, CycleSize entries, fixed by composition_
  VarKind* var_kinds_device_ =
      nullptr;  // device, n_vars_ entries, fixed by the problem
  std::vector<V*> node_buffers_;  // every op/Var node's buffer, owned here
  std::vector<unsigned char> feasible_host_;
  std::vector<T> obj_lb_host_;
  std::vector<V> candidate_point_host_;  // D2H-copied after each launch()
  T candidate_host_ =
      std::numeric_limits<T>::infinity();  // D2H-copied after each launch()
  std::int64_t candidate_index_host_ = -1;  // D2H-copied after each launch()
  Composition<CycleSize> composition_ {};
  std::size_t n_regions_ = 0;
  std::size_t n_elems_ = 0;

  std::size_t sample_points_ = 1;
  std::size_t slot_count_ = 0;
  std::size_t n_vars_ = 0;
  dim3 root_grid_ {};
  dim3 block_ {};
};

// Convenience aliases. Neither the fan-out widths nor the sample count are
// part of the type any more: both arrive as build() arguments, so these
// differ only in the value type V (and hence in which root kernel runs).
template<typename T, std::size_t CycleSize>
using IntervalGraphReplay = GraphReplay<T, cu::interval<T>, CycleSize>;

template<typename T, std::size_t CycleSize>
using PointGraphReplay = GraphReplay<T, T, CycleSize>;

// Deterministic evaluation over a fully-enumerable Composition: every region
// is an exact grid point rather than a random sample, so its CUB ArgMin
// reduction gives the true minimum objective over every point the
// Composition enumerates -- not a bound, the answer. GraphDriver only builds
// and uses one of these for Compositions where is_fully_enumerable() holds,
// and only dispatches to it for a node once every live variable in the box
// fits in its CycleSize slots (see GraphDriver::solve()).
template<typename T, std::size_t CycleSize>
using ExactGraphReplay = GraphReplay<T, T, CycleSize, /*Exact=*/true>;

}  // namespace cuminlp::dag
