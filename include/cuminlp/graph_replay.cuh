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
    const int* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind,
    cu::interval<T>* const* __restrict__ var_buffers,  // NUM_VARS x NUM_REGIONS
                                                       // (box bounds per
                                                       // variable per region)
    std::size_t n_vars,
    std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t r = tid; r < n_regions; r += num_threads) {
    auto ctx = partition::make_slot_context<CycleSize>(
        r, slot_var_ids, slot_fan_out, slot_kind);
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
    const int* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind,
    T* const* __restrict__ var_buffers,  // NUM_VARS x NUM_REGIONS
    std::size_t n_vars,
    std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t r = tid; r < n_regions; r += num_threads) {
    auto ctx = partition::make_slot_context<CycleSize>(
        r, slot_var_ids, slot_fan_out, slot_kind);
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
 * @tparam T numerical precision
 * @tparam CycleSize number of variables being cycled
 * @tparam SamplePoints number of points sampled per subdomain
 * @param parent_domain interval domain being divided
 * @param slot_var_ids  which variable each of the CycleSize slots acts on
 * @param slot_fan_out  fan-out (radix) of each slot
 * @param slot_kind     operation each slot performs
 * @param var_kinds     per-variable kind (Continuous samples get a uniform
 *                      real value; Integer/Binary get a uniform integer)
 * @param var_buffers buffer for sampled points
 * @param n_vars
 * @param n_regions
 * @param salt per-launch seed component, so re-visiting a box draws fresh
 * points
 */
template<typename T, std::size_t CycleSize, std::size_t SamplePoints>
__global__ void sample_points_kernel(
    const cu::interval<T>* __restrict parent_domain,
    const std::size_t* __restrict__ slot_var_ids,
    const int* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind,
    const VarKind* __restrict__ var_kinds,
    T* const* __restrict__ var_buffers,  // NUM_VARS x NUM_REGIONS x
                                         // SamplePoints
    std::size_t n_vars,
    std::size_t n_regions,
    std::size_t salt)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t r = tid; r < n_regions; r += num_threads) {
    auto ctx = partition::make_slot_context<CycleSize>(
        r, slot_var_ids, slot_fan_out, slot_kind);
    for (std::size_t vid = 0; vid < n_vars; ++vid) {
      cu::interval<T> v;
      partition::get_slot_bounds(ctx, parent_domain, vid, v);
      bool discrete = var_kinds[vid] != VarKind::Continuous;
      for (std::size_t i = 0; i < SamplePoints; ++i) {
        T p = discrete
            ? sample_discrete_from_interval(v.lb, v.ub, vid, i, r, salt)
            : sample_from_interval(v.lb, v.ub, vid, i, r, salt);
        var_buffers[vid][r * SamplePoints + i] = p;
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
__device__ T ipow_scalar(T base, int exponent)
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

// ipow_value dispatches to interval pown() or the scalar ipow_scalar()
// depending on which V this graph is templated on.
template<typename T>
__device__ cu::interval<T> ipow_value(cu::interval<T> a, int exponent)
{
  return pown(a, exponent);
}

template<typename T>
__device__ T ipow_value(T a, int exponent)
{
  return ipow_scalar(a, exponent);
}

template<typename T, typename V>
__global__ void ipow_kernel(const V* __restrict__ a,
                            int exponent,
                            V* __restrict__ out,
                            std::size_t n_regions)
{
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = ipow_value(a[i], exponent);
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
template<typename T,
         typename V,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints = 1,
         bool Exact = false>
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
   */
  GraphBuilder(const Problem<T>& problem,
               cu::interval<T>* domain_buffer,
               std::size_t n_regions,
               const std::size_t* slot_var_ids,
               const int* slot_fan_out,
               const SlotKind* slot_kind,
               const VarKind* var_kinds)
      : problem_(problem)
      , n_regions_(n_regions)
      , n_elems_(
            is_point
                ? n_regions * SamplePoints
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
                                  n_regions_);
    } else if constexpr (is_point) {
      root_node_ = detail::add_kernel_node(
          graph_,
          {},
          sample_points_kernel<T, CycleSize, SamplePoints>,
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
                                  n_regions_);
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
      case Op::IPow:
        producer_nodes_[id] = wire_ipow(id);
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

  cudaGraphNode_t wire_ipow(std::size_t id)
  {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    if (node.op != Op::IPow) {
      throw std::runtime_error(
          "wire_ipow called on a node whose op is not Op::IPow");
    }
    if (node.in.size() != 1) {
      throw std::runtime_error(
          "wire_ipow called on a node without exactly one operand");
    }
    std::size_t operand_id = node.in[0];
    if (problem_.graph.nodes[operand_id].op == Op::Const) {
      throw std::runtime_error("Op::IPow applied directly to an Op::Const; unreachable via the "
                               "current Expr API");
    }
    cudaGraphNode_t operand_node = ensure_node(operand_id);
    buffers_[id] = detail::alloc_device<V>(n_elems_);
    return detail::add_kernel_node(graph_,
                                   {operand_node},
                                   ipow_kernel<T, V>,
                                   grid_,
                                   block_,
                                   buffers_[operand_id],
                                   node.payload.int_exp,
                                   buffers_[id],
                                   n_elems_);
  }

  const Problem<T>& problem_;
  std::size_t n_regions_;
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
template<typename T,
         typename V,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints = 1,
         std::size_t EnumerateCap = PartitionNum,
         bool Exact = false>
class GraphReplay
{
  static constexpr bool is_point = std::is_same_v<V, T>;

public:
  // `composition` fixes this replay's fan-out/shape for its whole lifetime;
  // only which variables fill its slots (set_domain's `var_ids`) varies
  // between launches.
  static GraphReplay build(const Problem<T>& problem,
                           const Composition<CycleSize>& composition)
  {
    GraphReplay replay;
    replay.composition_ = composition;
    replay.n_regions_ =
        composition_fan_out<PartitionNum, EnumerateCap>(composition);
    replay.n_vars_ = problem.box_bounds.size();

    replay.domain_buffer_ =
        detail::alloc_device<cu::interval<T>>(replay.n_vars_);
    replay.slot_var_ids_device_ = detail::alloc_device<std::size_t>(CycleSize);
    replay.slot_fan_out_device_ = detail::alloc_device<int>(CycleSize);
    replay.slot_kind_device_ = detail::alloc_device<SlotKind>(CycleSize);
    replay.var_kinds_device_ = detail::alloc_device<VarKind>(replay.n_vars_);

    // fan_out/kind are fixed by `composition`, and var_kinds by the problem,
    // so they're all uploaded once here rather than on every set_domain()
    // call like var_ids.
    std::array<int, CycleSize> fan_out_host {};
    for (std::size_t j = 0; j < CycleSize; ++j) {
      fan_out_host[j] = static_cast<int>(
          slot_fan_out<PartitionNum, EnumerateCap>(composition[j]));
    }
    detail::check(cudaMemcpy(replay.slot_fan_out_device_,
                             fan_out_host.data(),
                             CycleSize * sizeof(int),
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

    GraphBuilder<T, V, CycleSize, PartitionNum, SamplePoints, Exact> builder(
        problem,
        replay.domain_buffer_,
        replay.n_regions_,
        replay.slot_var_ids_device_,
        replay.slot_fan_out_device_,
        replay.slot_kind_device_,
        replay.var_kinds_device_);

    // n_elems_ is n_regions_ for the interval graph, n_regions_ * SamplePoints
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
          &salt,
      };
      params.func = reinterpret_cast<void*>(
          sample_points_kernel<T, CycleSize, SamplePoints>);
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

  // Flat element index the reduction picked; r * SamplePoints + i for the
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
  int* slot_fan_out_device_ =
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
  std::size_t n_vars_ = 0;
  dim3 root_grid_ {};
  dim3 block_ {};
};

// Convenience aliases: the interval graph doesn't need SamplePoints, the
// point graph does. EnumerateCap defaults to PartitionNum, same as GraphReplay
// itself and GreedyCompositionPolicy -- pass it explicitly only when
// decoupling the enumerate threshold from the bisection width.
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t EnumerateCap = PartitionNum>
using IntervalGraphReplay =
    GraphReplay<T, cu::interval<T>, CycleSize, PartitionNum, 1, EnumerateCap>;

template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints,
         std::size_t EnumerateCap = PartitionNum>
using PointGraphReplay =
    GraphReplay<T, T, CycleSize, PartitionNum, SamplePoints, EnumerateCap>;

// Deterministic evaluation over a fully-enumerable Composition: every region
// is an exact grid point rather than a random sample, so its CUB ArgMin
// reduction gives the true minimum objective over every point the
// Composition enumerates -- not a bound, the answer. GraphDriver only builds
// and uses one of these for Compositions where is_fully_enumerable() holds,
// and only dispatches to it for a node once every live variable in the box
// fits in its CycleSize slots (see GraphDriver::solve()).
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t EnumerateCap = PartitionNum>
using ExactGraphReplay =
    GraphReplay<T, T, CycleSize, PartitionNum, 1, EnumerateCap, /*Exact=*/true>;

}  // namespace cuminlp::dag
