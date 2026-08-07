#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
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
//
// `widths` is parallel to `composition.kinds`/`var_ids` too
// (design/BUDGETED_PARTITION.md §4): the fan-out slot `j` actually decodes
// against, as chosen by the policy for *this* box -- no longer derivable
// from `composition[j]` alone the way a shared FanOutSpec made it under the
// old uniform-width policies. It is per-launch data, never part of graph
// identity (§4.1): two SlotAssignments with the same `composition` but
// different `widths` share one cached graph, exactly as they already share
// one graph despite different `var_ids`.
//
// `domain_sizes` is empty unless a policy rounds an enumerate slot's width
// up past its true domain (§8): when non-empty it is parallel to `widths`,
// and `domain_sizes[j] < widths[j]` marks slot `j` as carrying duplicate
// parts in `[domain_sizes[j], widths[j])` (§8.1) -- `domain_sizes[j] ==
// widths[j]` (no rounding) is the common case and never flags a duplicate.
struct SlotAssignment
{
  Composition composition;
  std::vector<std::size_t> var_ids;
  std::vector<std::size_t> widths;
  std::vector<std::size_t> domain_sizes;

  std::size_t size() const { return composition.size(); }
};

/**
 * @brief A 64-bit hash of everything a decode needs: `(kinds, var_ids,
 * widths)`.
 *
 * The purity tripwire (design/BUDGETED_PARTITION.md §9): with widths now
 * data-dependent rather than a function of kind alone, two SlotAssignments
 * with the same *slot count* can decode `sidx` into completely different,
 * entirely plausible-looking boxes -- comparing slot counts (as the old
 * tripwire did) cannot see that. `search::Node` stores this hash when it
 * enqueues a child and compares it against a fresh call to
 * `assignment_hash(policy.choose(parent, var_kinds))` when it re-derives
 * that child's box; a policy that is not a pure function of `(box,
 * var_kinds, calibration)` produces a different assignment and hence a
 * different hash, caught here rather than decoded silently.
 *
 * Not cryptographic -- FNV-1a over the three parallel arrays -- since the
 * only property needed is "different assignments almost certainly hash
 * differently within one process run", not resistance to a deliberate
 * collision.
 */
inline std::uint64_t assignment_hash(const SlotAssignment& assignment)
{
  std::uint64_t h = 1469598103934665603ull;  // FNV-1a 64-bit offset basis
  auto mix = [&h](std::uint64_t v)
  {
    h ^= v;
    h *= 1099511628211ull;  // FNV-1a 64-bit prime
  };
  for (SlotKind k : assignment.composition) {
    mix(static_cast<std::uint64_t>(k));
  }
  for (std::size_t v : assignment.var_ids) {
    mix(static_cast<std::uint64_t>(v));
  }
  for (std::size_t w : assignment.widths) {
    mix(static_cast<std::uint64_t>(w));
  }
  return h;
}

/// Prefix products of `widths` (design/MODULE_REFACTOR.md §4.2): prefix[j] =
/// product of widths[0..j-1] (empty product 1 for j == 0), so slot j's digit
/// for region r is `(r / prefix[j]) % widths[j]`. Saturates like
/// composition_fan_out below. The widths-taking counterpart of
/// slot_prefixes(Composition, FanOutSpec) -- this one asks nothing about
/// slot kind, so it works for any per-slot width vector, uniform or not.
inline std::vector<std::size_t> slot_prefixes(
    std::span<const std::size_t> widths)
{
  std::vector<std::size_t> prefix(widths.size());
  std::size_t running = 1;
  for (std::size_t j = 0; j < widths.size(); ++j) {
    prefix[j] = running;
    std::size_t const width = widths[j];
    if (width != 0 && running > std::numeric_limits<std::size_t>::max() / width)
    {
      running = std::numeric_limits<std::size_t>::max();
    } else {
      running *= width;
    }
  }
  return prefix;
}

/// Product of `widths`, saturating at SIZE_MAX rather than wrapping -- the
/// widths-taking counterpart of composition_fan_out(Composition, FanOutSpec)
/// below, for a per-slot width vector that isn't a uniform function of kind.
inline std::size_t composition_fan_out(std::span<const std::size_t> widths)
{
  std::size_t total = 1;
  for (std::size_t width : widths) {
    if (width != 0 && total > std::numeric_limits<std::size_t>::max() / width) {
      return std::numeric_limits<std::size_t>::max();
    }
    total *= width;
  }
  return total;
}

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

/**
 * @brief True iff region `r` is a duplicate child under `assignment`
 *        (design/BUDGETED_PARTITION.md §8.1).
 *
 * A duplicate is a region whose digit, for some slot, falls in
 * `[domain_sizes[j], widths[j])` -- the padding a rounded-up enumerate
 * width (§8) or leftover-budget padding (§7's fallback) introduces. Such a
 * region decodes to the same box as its `domain_sizes[j] - 1` sibling, so
 * enqueueing it would explore the same subtree twice.
 *
 * `prefix` must be `slot_prefixes(assignment.widths)`; passed in rather than
 * recomputed here, since a caller checking every child of one node computes
 * it once and reuses it per region.
 *
 * Always false when `assignment.domain_sizes` is empty: only
 * BisectionBudgetCompositionPolicy populates it, so a composition from
 * GreedyEnumerate/WidthFirst never has a duplicate child by this
 * definition.
 */
inline bool is_duplicate_child(std::size_t r,
                               const SlotAssignment& assignment,
                               std::span<const std::size_t> prefix)
{
  if (assignment.domain_sizes.empty()) {
    return false;
  }
  for (std::size_t j = 0; j < assignment.composition.size(); ++j) {
    std::size_t const part = (r / prefix[j]) % assignment.widths[j];
    if (part >= assignment.domain_sizes[j]) {
      return true;
    }
  }
  return false;
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
