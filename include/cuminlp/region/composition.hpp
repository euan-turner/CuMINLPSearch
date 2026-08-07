#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/region/slot.hpp"

namespace cuminlp::region
{

// What each slot does this iteration, one entry per live slot -- no
// compile-time capacity and no padding tail (design/MODULE_REFACTOR.md §4.6):
// a composition is exactly its live slots, so `size()` is the only length
// there is.
struct Composition
{
  std::vector<SlotKind> kinds;

  std::size_t size() const { return kinds.size(); }

  bool operator==(const Composition&) const = default;

  constexpr auto begin() const { return kinds.begin(); }

  constexpr auto end() const { return kinds.end(); }

  SlotKind operator[](std::size_t i) const { return kinds[i]; }

  SlotKind& operator[](std::size_t i) { return kinds[i]; }

  const SlotKind* data() const { return kinds.data(); }
};

// Which variable fills each slot of `composition`, for one search-tree node.
// `var_ids` is parallel to `composition.kinds` and must be pairwise distinct,
// each indexing a live (lb < ub) dimension -- CompositionPolicy's contract
// (see policy/policy.hpp, and §4.5).
struct SlotAssignment
{
  Composition composition;
  std::vector<std::size_t> var_ids;

  std::size_t size() const { return composition.size(); }
};

// Total children a Composition produces -- the product of its slots'
// fan-outs, i.e. the `n_regions` a GraphReplay built for it will launch.
//
// Saturates at SIZE_MAX rather than wrapping. With partition_num a runtime
// value, e.g. 20 slots at partition_num 10 (10^20, past the 1.8e19 size_t
// ceiling) is reachable from a command line, and a wrapped product would
// silently size buffers *too small* instead of failing. Callers size
// allocations off this, so saturation turns that into an obvious
// ResourceExhausted at GraphReplay::build() -- see estimate_bytes().
inline std::size_t composition_fan_out(const Composition& composition,
                                       const FanOutSpec& fan_out)
{
  std::size_t total = 1;
  for (SlotKind kind : composition) {
    std::size_t const width = slot_fan_out(kind, fan_out);
    if (width != 0 && total > std::numeric_limits<std::size_t>::max() / width) {
      return std::numeric_limits<std::size_t>::max();
    }
    total *= width;
  }
  return total;
}

// Prefix products for a Composition's slots (design/MODULE_REFACTOR.md §4.2):
// prefix[j] = product of fan_out[0..j-1] (empty product 1 for j == 0), so
// slot j's digit for region r is `(r / prefix[j]) % fan_out[j]` -- the same
// digit a per-thread repeated-division loop computes, obtained
// arithmetically and uploaded once per Composition instead of rebuilt by
// every thread. Saturates like composition_fan_out.
inline std::vector<std::size_t> slot_prefixes(const Composition& composition,
                                              const FanOutSpec& fan_out)
{
  std::vector<std::size_t> prefix(composition.size());
  std::size_t running = 1;
  for (std::size_t j = 0; j < composition.size(); ++j) {
    prefix[j] = running;
    std::size_t const width = slot_fan_out(composition[j], fan_out);
    if (width != 0 && running > std::numeric_limits<std::size_t>::max() / width)
    {
      running = std::numeric_limits<std::size_t>::max();
    } else {
      running *= width;
    }
  }
  return prefix;
}

// True iff every slot enumerates (IntegerEnumerate/BinaryEnumerate) rather
// than partitioning or ranging over a live Continuous variable. Combined
// with a node's live-variable count fitting in the slots the policy
// actually filled (checked separately, since that's a property of a box,
// not of a Composition in isolation), this is exactly the condition under
// which every child this Composition produces is a fully-resolved point --
// see SearchDriver's use of the backend's enumerator role.
inline bool is_fully_enumerable(const Composition& composition)
{
  for (SlotKind kind : composition) {
    if (kind != SlotKind::IntegerEnumerate && kind != SlotKind::BinaryEnumerate)
    {
      return false;
    }
  }
  return true;
}

// A node may be fathomed without enqueueing children iff every live variable
// fits in the slots this iteration actually filled and the resulting
// Composition is fully enumerable (TEST_EXTENSION.md). Pure function of a
// box's live-variable count and the Composition the policy chose for it, so
// it's testable independent of the surrounding search loop (see
// search/driver.hpp).
inline bool can_fathom_without_children(std::size_t live_count,
                                        const Composition& composition)
{
  return live_count <= composition.size() && is_fully_enumerable(composition);
}

// A composition's compact spelling, one slot_kind_char per slot -- the form
// diagnostics use (e.g. the GRAPH line, design/TELEMETRY.md §4.2). Lives here
// rather than in backend::graph because a composition's spelling is region's
// to define, and a second backend would want the same one.
inline std::string spell(const Composition& composition)
{
  std::string out(composition.size(), '?');
  for (std::size_t i = 0; i < composition.size(); ++i) {
    out[i] = slot_kind_char(composition[i]);
  }
  return out;
}

}  // namespace cuminlp::region
