#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuinterval/cuinterval.h>
#include <cuinterval/interval.h>

#include <cuda/std/limits>

#include "cuminlp/backend/graph/ops.cuh"
#include "cuminlp/model/dag.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/decode.hpp"

// The DAG-evaluation kernels: the two-pass root materialisation
// (broadcast_domain_kernel/apply_slots_kernel and their exact-point
// counterparts, design/MODULE_REFACTOR.md §4.2-4.3) plus the generic
// per-node op-evaluation kernels GraphBuilder wires one instance of per DAG
// node. Moved out of graph_replay.cuh (§11).
namespace cuminlp::backend::graph
{

// Unqualified names below (SlotKind, Composition, VarKind, Op::..., etc.)
// are the same ones graph_replay.cuh used unqualified before this file
// existed, when they lived in flat cuminlp:: (composition_policy.hpp,
// dag.hpp) and this content sat in namespace cuminlp::dag -- enclosing-scope
// lookup found them without qualification. Now that they live in sibling
// namespaces cuminlp::region / cuminlp::model, that lookup no longer reaches
// them, so these using-declarations restore it explicitly rather than
// re-qualifying every use-site (design/MODULE_REFACTOR.md §11 is a pure
// rename; this keeps the diff a namespace/file move, not a rewrite).
using cuminlp::region::can_fathom_without_children;
using cuminlp::region::Composition;
using cuminlp::region::composition_fan_out;
using cuminlp::region::FanOutSpec;
using cuminlp::region::is_fully_enumerable;
using cuminlp::region::slot_fan_out;
using cuminlp::region::slot_prefixes;
using cuminlp::region::SlotAssignment;
using cuminlp::region::SlotKind;
namespace decode = cuminlp::region::decode;
using cuminlp::model::Cmp;
using cuminlp::model::ConstraintRef;
using cuminlp::model::DAGNode;
using cuminlp::model::ExprDAG;
using cuminlp::model::Op;
using cuminlp::model::Problem;
using cuminlp::model::VarKind;

// Two-pass region materialisation (design/MODULE_REFACTOR.md §4.2-4.3),
// replacing a per-thread register-resident slot context and an
// O(n_vars x slot_count) scan with two grid-parallel kernels:
//
//   broadcast_domain_kernel*  every (variable, region[, sample]) starts at
//                             the parent box's value for that variable.
//   apply_slots_kernel*       for each (live slot, region), decodes that
//                             slot's digit arithmetically from the region
//                             index (`(r / prefix[j]) % fan_out[j]`, the same
//                             digit a repeated-division loop computes -- see
//                             slot_prefixes in region/composition.hpp) and
//                             overwrites exactly the (variable, region) pairs
//                             that slot narrows.
//
// The two run with an explicit graph dependency (apply after broadcast), so
// the overwrite cannot race the broadcast. Distinct var_ids across slots
// (CompositionPolicy's contract) is what makes apply's "last write wins"
// agree with the old context's "first match wins": each dimension is
// touched by at most one slot. Both kernels are template-free in any
// capacity -- slot_count is a runtime argument, not an array bound.

/// Broadcasts the parent box into every region's interval buffer.
template<typename T>
__global__ void broadcast_domain_kernel(
    const cu::interval<T>* __restrict__ parent_domain,
    cu::interval<T>* const* __restrict__ var_buffers,
    std::size_t n_vars,
    std::size_t n_regions)
{
  std::size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = gridDim.x * blockDim.x;
  std::size_t const total = n_vars * n_regions;
  for (std::size_t idx = tid; idx < total; idx += stride) {
    std::size_t const vid = idx / n_regions;
    var_buffers[vid][idx % n_regions] = parent_domain[vid];
  }
}

/// Narrows each live slot's region-specific bound, overwriting what
/// broadcast_domain_kernel wrote for that (variable, region) pair.
template<typename T>
__global__ void apply_slots_kernel(
    const cu::interval<T>* __restrict__ parent_domain,
    const std::size_t* __restrict__ slot_var_ids,
    const std::uint32_t* __restrict__ slot_fan_out,
    const std::size_t* __restrict__ slot_prefix,
    const SlotKind* __restrict__ slot_kind,
    cu::interval<T>* const* __restrict__ var_buffers,
    std::size_t n_regions,
    std::size_t slot_count)
{
  std::size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = gridDim.x * blockDim.x;
  std::size_t const total = slot_count * n_regions;
  for (std::size_t idx = tid; idx < total; idx += stride) {
    std::size_t const j = idx / n_regions;
    std::size_t const r = idx % n_regions;
    std::size_t const part = (r / slot_prefix[j]) % slot_fan_out[j];
    std::size_t const vid = slot_var_ids[j];
    cu::interval<T> v;
    decode::slot_bounds<T>(
        slot_kind[j], parent_domain[vid], part, slot_fan_out[j], v);
    var_buffers[vid][r] = v;
  }
}

/**
 * @brief Deterministic counterpart, for a fully-enumerable Composition
 * (every slot IntegerEnumerate/BinaryEnumerate, SearchDriver's precondition
 * for using this pair): broadcasts each variable's exact point value
 * (parent_domain[vid].lb -- a non-live dimension is degenerate, so lb is
 * the exact point). Writes T rather than cu::interval<T>, so it plugs into
 * the same V=T op-kernel pipeline (feasibility_check_kernel,
 * objective_extract_kernel, CUB ArgMin) the sampling point graph feeds --
 * this reuses that pipeline as-is to get the true minimum objective over
 * every point in the enumeration, not an interval relaxation.
 */
template<typename T>
__global__ void broadcast_domain_point_kernel(
    const cu::interval<T>* __restrict__ parent_domain,
    T* const* __restrict__ var_buffers,
    std::size_t n_vars,
    std::size_t n_regions)
{
  std::size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = gridDim.x * blockDim.x;
  std::size_t const total = n_vars * n_regions;
  for (std::size_t idx = tid; idx < total; idx += stride) {
    std::size_t const vid = idx / n_regions;
    var_buffers[vid][idx % n_regions] = parent_domain[vid].lb;
  }
}

/// Enumerate counterpart of apply_slots_kernel: every slot decodes to an
/// exact point (lb == ub), so the narrowed bound's lb is taken directly.
template<typename T>
__global__ void apply_slots_point_kernel(
    const cu::interval<T>* __restrict__ parent_domain,
    const std::size_t* __restrict__ slot_var_ids,
    const std::uint32_t* __restrict__ slot_fan_out,
    const std::size_t* __restrict__ slot_prefix,
    const SlotKind* __restrict__ slot_kind,
    T* const* __restrict__ var_buffers,
    std::size_t n_regions,
    std::size_t slot_count)
{
  std::size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = gridDim.x * blockDim.x;
  std::size_t const total = slot_count * n_regions;
  for (std::size_t idx = tid; idx < total; idx += stride) {
    std::size_t const j = idx / n_regions;
    std::size_t const r = idx % n_regions;
    std::size_t const part = (r / slot_prefix[j]) % slot_fan_out[j];
    std::size_t const vid = slot_var_ids[j];
    cu::interval<T> v;
    decode::slot_bounds<T>(
        slot_kind[j], parent_domain[vid], part, slot_fan_out[j], v);
    var_buffers[vid][r] = v.lb;  // exact point: v.lb == v.ub
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

}  // namespace cuminlp::backend::graph
