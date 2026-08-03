#pragma once

#include <cmath>
#include <cstddef>

#include <cuinterval/interval.h>

#include "cuminlp/composition_policy.hpp"

// __host__ __device__ only mean anything to nvcc; on a plain host compiler
// (e.g. building the host-only test targets without CUDA) they'd be
// undefined identifiers, so they're no-ops outside __CUDACC__. This lets the
// exact same function be called from partition.cuh's __device__ kernels and
// from search.hpp's host-only CompositionInterval::materialise, which is
// what makes host/device agreement structural rather than something to prove
// by test (see TEST_EXTENSION.md).
#if defined(__CUDACC__)
#  define CUMINLP_HD __host__ __device__
#else
#  define CUMINLP_HD
#endif

namespace cuminlp::decode
{

// Decodes one slot's contribution to a dimension's bounds: `part` is this
// slot's index into its `fan_out`-way radix (see
// partition::make_slot_context/make_cycle_context), `parent` the dimension's
// bounds before this slot acted on it.
//
// IntegerEnumerate/BinaryEnumerate snap to the integer lattice via
// ceil(parent.lb) before offsetting by `part`, rather than assuming
// parent.lb is already integer-valued: the normal case after an
// IntegerBisect is that it isn't, since IntegerBisect reuses the continuous
// linear-width formula (see TEST_EXTENSION.md). A Binary variable's bounds
// are always exactly integer already, so ceil is a no-op there. `value`
// exceeding `parent.ub` (fan_out wider than the true remaining domain, or a
// domain with no integer point at all) clamps to parent.ub -- sound
// (a clamped duplicate is still inside the parent domain), just possibly a
// duplicate evaluation.
template<typename T>
CUMINLP_HD inline void slot_bounds(SlotKind kind,
                                   const cu::interval<T>& parent,
                                   std::size_t part,
                                   std::size_t fan_out,
                                   cu::interval<T>& bound)
{
  switch (kind) {
    case SlotKind::IntegerEnumerate:
    case SlotKind::BinaryEnumerate: {
      T lo = std::ceil(parent.lb);
      T value = lo + static_cast<T>(part);
      if (value > parent.ub) {
        value = parent.ub;
      }
      bound.lb = value;
      bound.ub = value;
      break;
    }
    case SlotKind::Continuous:
    case SlotKind::IntegerBisect: {
      T width = (parent.ub - parent.lb) / static_cast<T>(fan_out);
      bound.lb = parent.lb + width * static_cast<T>(part);
      bound.ub = parent.lb + width * static_cast<T>(part + 1);
      break;
    }
    case SlotKind::Padding: {
      // fan_out is always 1 (part always 0): no live variable assigned to
      // this slot, so the dimension it would otherwise touch is untouched
      // by it.
      //
      // This branch is emphatically NOT unreachable, and a comment here once
      // claiming so hid a bug. GreedyCompositionPolicy fills a Padding slot's
      // var_id by *repeating the last variable it assigned*, so a Padding
      // slot's var_id routinely names a real, live dimension. Callers must
      // therefore not iterate the padding tail: `bound = parent` here undoes
      // any narrowing an earlier real slot applied to that same dimension.
      //
      // partition::get_slot_bounds is safe because it returns on its first
      // var_ids match; search::CompositionInterval::materialise is safe
      // because it stops at Composition::count. Any third caller needs to do
      // one or the other.
      bound = parent;
      break;
    }
  }
}

}  // namespace cuminlp::decode
