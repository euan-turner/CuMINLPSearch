#pragma once

#include <cmath>
#include <cstddef>

#include <cuinterval/interval.h>

#include "cuminlp/region/slot.hpp"

#if defined(__CUDACC__)
#  define CUMINLP_HD __host__ __device__
#else
#  define CUMINLP_HD
#endif

namespace cuminlp::region::decode
{

// Decodes one slot's contribution to a dimension's bounds: `part` is this
// slot's index into its `fan_out`-way radix (see slot_prefixes in
// region/composition.hpp, design/MODULE_REFACTOR.md §4.2), `parent` the
// dimension's bounds before this slot acted on it.
//
// IntegerEnumerate/BinaryEnumerate snap to the integer lattice via
// ceil(parent.lb) before offsetting by `part`, rather than assuming
// parent.lb is already integer-valued: the normal case after an
// IntegerPartition is that it isn't, since IntegerPartition reuses the
// continuous linear-width formula (see TEST_EXTENSION.md). A Binary
// variable's bounds are always exactly integer already, so ceil is a no-op
// there. `value` exceeding `parent.ub` (fan_out wider than the true
// remaining domain, or a domain with no integer point at all) clamps to
// parent.ub -- sound (a clamped duplicate is still inside the parent
// domain), just possibly a duplicate evaluation.
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
    case SlotKind::IntegerPartition: {
      T width = (parent.ub - parent.lb) / static_cast<T>(fan_out);
      bound.lb = parent.lb + width * static_cast<T>(part);
      bound.ub = parent.lb + width * static_cast<T>(part + 1);
      break;
    }
  }
}

}  // namespace cuminlp::region::decode
