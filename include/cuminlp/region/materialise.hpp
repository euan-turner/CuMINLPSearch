#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/decode.hpp"
#include "cuminlp/region/fan_out.hpp"

// The single decode `search::Node::materialise` and the device's
// apply_slots_kernel both go through: given a
// parent box and the SlotAssignment/sidx that produced a child, reconstruct
// the child's box. Knows nothing about a policy, a device or search state --
// the caller (search::Node::materialise) is what re-derives `assignment` by
// calling CompositionPolicy::choose() and owns the purity tripwire; this is
// just the arithmetic once that assignment is in hand.
namespace cuminlp::region
{

template<typename T>
void materialise(std::span<const cu::interval<T>> parent,
                 std::size_t sidx,
                 const SlotAssignment& assignment,
                 const FanOutSpec& fan_out,
                 std::vector<cu::interval<T>>& out)
{
  out.assign(parent.begin(), parent.end());

  std::size_t idx = sidx;
  for (std::size_t j = 0; j < assignment.composition.size(); ++j) {
    SlotKind kind = assignment.composition[j];
    std::size_t width = slot_fan_out(kind, fan_out);
    std::size_t part = idx % width;
    idx /= width;

    std::size_t dim = assignment.var_ids[j];
    decode::slot_bounds<T>(kind, parent[dim], part, width, out[dim]);
  }
}

}  // namespace cuminlp::region
