#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuinterval/interval.h>

#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/decode.hpp"

// The subregion sampler's kernels: draws `sample_points` samples per
// (variable, region) from each region's own narrowed bound, salted per
// iteration -- today's only RegionSampler implementation, composition-
// coupled (design/MODULE_REFACTOR.md §5.2 decouples this; that is stage 8,
// not this move). Moved out of graph_replay.cuh (§11); the filename is
// this backend's eventual sibling to uniform_sampler.cuh once §5.2 lands.
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
// partitioning an integer variable whose domain was too wide to enumerate
// outright (see GreedyCompositionPolicy) -- IntegerPartition reuses the same
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
 * @brief Broadcast pass for the sampling point graph: draws `sample_points`
 * samples per (variable, region) directly from the parent box -- the value
 * every region starts with before any live slot narrows it. Every
 * (variable, region, sample-index) triple is an independent thread, unlike
 * apply_slots_sample_kernel below which threads over (slot, region) and
 * loops sample_points internally.
 *
 * `sample_points` is a kernel argument rather than a template parameter,
 * same as before the two-pass split.
 */
template<typename T>
__global__ void broadcast_domain_sample_kernel(
    const cu::interval<T>* __restrict__ parent_domain,
    const VarKind* __restrict__ var_kinds,
    T* const* __restrict__ var_buffers,  // NUM_VARS x NUM_REGIONS x
                                         // sample_points
    std::size_t n_vars,
    std::size_t n_regions,
    std::size_t sample_points,
    std::size_t salt)
{
  std::size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = gridDim.x * blockDim.x;
  std::size_t const total = n_vars * n_regions * sample_points;
  for (std::size_t idx = tid; idx < total; idx += stride) {
    std::size_t const per_var = n_regions * sample_points;
    std::size_t const vid = idx / per_var;
    std::size_t const rem = idx % per_var;
    std::size_t const r = rem / sample_points;
    std::size_t const i = rem % sample_points;
    cu::interval<T> const v = parent_domain[vid];
    bool const discrete = var_kinds[vid] != VarKind::Continuous;
    T const p = discrete
        ? sample_discrete_from_interval(v.lb, v.ub, vid, i, r, salt)
        : sample_from_interval(v.lb, v.ub, vid, i, r, salt);
    var_buffers[vid][r * sample_points + i] = p;
  }
}

/// Sample counterpart of apply_slots_kernel: re-draws each live slot's
/// sample_points from its narrowed bound, overwriting the broadcast pass's
/// draw from the parent bound for that (variable, region) pair -- last
/// write wins (design/MODULE_REFACTOR.md §4.4).
template<typename T>
__global__ void apply_slots_sample_kernel(
    const cu::interval<T>* __restrict__ parent_domain,
    const std::size_t* __restrict__ slot_var_ids,
    const std::uint32_t* __restrict__ slot_fan_out,
    const std::size_t* __restrict__ slot_prefix,
    const SlotKind* __restrict__ slot_kind,
    const VarKind* __restrict__ var_kinds,
    T* const* __restrict__ var_buffers,
    std::size_t n_regions,
    std::size_t slot_count,
    std::size_t sample_points,
    std::size_t salt)
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
    bool const discrete = var_kinds[vid] != VarKind::Continuous;
    for (std::size_t i = 0; i < sample_points; ++i) {
      T const p = discrete
          ? sample_discrete_from_interval(v.lb, v.ub, vid, i, r, salt)
          : sample_from_interval(v.lb, v.ub, vid, i, r, salt);
      var_buffers[vid][r * sample_points + i] = p;
    }
  }
}

}  // namespace cuminlp::backend::graph
