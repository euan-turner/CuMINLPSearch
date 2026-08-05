#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"

namespace cuminlp
{

// Which operation a slot in a Composition performs on its assigned variable
// this iteration. This is what fixes a replayed CUDA graph's fan-out/shape,
// so each distinct sequence of SlotKinds corresponds to its own pre-built
// graph.
enum class SlotKind : std::uint8_t
{
  Continuous,
  IntegerPartition,
  IntegerEnumerate,
  BinaryEnumerate,
};

inline const char* slot_kind_name(SlotKind kind)
{
  switch (kind) {
    case SlotKind::Continuous:        return "Continuous";
    case SlotKind::IntegerPartition:  return "IntegerPartition";
    case SlotKind::IntegerEnumerate:  return "IntegerEnumerate";
    case SlotKind::BinaryEnumerate:   return "BinaryEnumerate";
  }
  return "?";
}

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
// (see below, and §4.5).
struct SlotAssignment
{
  Composition composition;
  std::vector<std::size_t> var_ids;

  std::size_t size() const { return composition.size(); }
};

// How wide each slot fans out. Formerly the PartitionNum/EnumerateCap
// template parameters; they never reached device codegen (the per-slot
// fan-outs they produce are computed host-side and uploaded to a device
// buffer, see GraphReplay::build), so nothing about CUDA graph capture
// required them to be compile-time constants -- only C++ did, and only
// incidentally. As runtime values a single binary can tune them per problem.
//
// The compiler no longer rejects a nonsensical value, so the constructor
// does: partition_num < 2 is the important one, since a fan-out of 1
// subdivides nothing and would let the search loop forever without ever
// narrowing a box.
class FanOutSpec
{
public:
  FanOutSpec(std::size_t partition_num, std::size_t enumerate_cap)
      : partition_num_(partition_num)
      , enumerate_cap_(enumerate_cap)
  {
    if (partition_num < 2) {
      throw InvalidConfiguration(
          "partition_num must be at least 2; a fan-out of 1 subdivides "
          "nothing, so bisection could never narrow a box");
    }
    if (enumerate_cap < 1) {
      throw InvalidConfiguration(
          "enumerate_cap must be at least 1; a slot must produce at least "
          "one child");
    }
  }

  // EnumerateCap used to default to PartitionNum as a template parameter;
  // this keeps that shorthand for callers that don't want the distinction.
  explicit FanOutSpec(std::size_t partition_num)
      : FanOutSpec(partition_num, partition_num)
  {
  }

  std::size_t partition_num() const { return partition_num_; }
  std::size_t enumerate_cap() const { return enumerate_cap_; }

  bool operator==(const FanOutSpec&) const = default;

private:
  std::size_t partition_num_;
  std::size_t enumerate_cap_;
};

// Number of children a slot of the given kind produces. Binary is always
// exactly 2 (its domain is always size 2 until resolved); Continuous/
// IntegerPartition share partition_num (the partition width); IntegerEnumerate
// uses its own enumerate_cap, independent of partition_num, so raising the
// enumerate threshold doesn't force wider partitioning elsewhere.
inline std::size_t slot_fan_out(SlotKind kind, const FanOutSpec& fan_out)
{
  if (kind == SlotKind::BinaryEnumerate) {
    return 2;
  }
  if (kind == SlotKind::IntegerEnumerate) {
    return fan_out.enumerate_cap();
  }
  return fan_out.partition_num();
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
    if (width != 0 && running > std::numeric_limits<std::size_t>::max() / width) {
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
// see GraphDriver's use of ExactGraphReplay.
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
// graph_driver.cuh).
inline bool can_fathom_without_children(std::size_t live_count,
                                        const Composition& composition)
{
  return live_count <= composition.size() && is_fully_enumerable(composition);
}

// Hardware- and user-derived tuning inputs, read once and then frozen for
// the lifetime of a solve.
//
// The freezing is a *correctness* requirement, not a performance choice.
// search::CompositionInterval::materialise reconstructs a node's box by
// re-invoking policy.choose() on the parent box and decoding sidx against
// the fan-outs that produces. If choose() consulted live device state --
// free memory, occupancy, queue depth -- then the assignment at enqueue time
// and at materialise time could differ, sidx would decode against the wrong
// radix vector, and the search would compute silently wrong boxes: no crash,
// no exception, a plausible-looking wrong optimum. Freezing at solve() entry
// lets the policy adapt to the hardware while staying a pure function of
// (box, var_kinds, calibration). See design/RUNTIME_SHAPE.md.
struct SearchCalibration
{
  // Upper bound on slots the policy may fill. Never above kMaxSlots.
  std::size_t max_cycle_size = 0;
  std::size_t free_device_bytes = 0;
  std::size_t multiprocessor_count = 0;
};

// Search-shape guard, not a compiled bound: beyond ~64 slots a single
// composition's fan-out product is past anything a device can hold at any
// fan-out, so this is a ceiling on absurdity, in the same sense
// partition_ceiling (policy_catalogue.hpp) is (design/MODULE_REFACTOR.md
// §4.7).
inline constexpr std::size_t kMaxSlots = 64;

// Decides, for a search-tree node's current box, which variables to act on
// next and how many slots to use. Must be a pure function of the box, the
// problem's variable kinds, and the frozen SearchCalibration -- nothing that
// can change during a run (see SearchCalibration above).
//
// The policy also *owns* the FanOutSpec. This is deliberate: decoding a
// node's box (search::CompositionInterval::materialise) needs both the
// SlotAssignment and the fan-outs its sidx was encoded against, and reading
// both off one object makes them impossible to disagree.
//
// `choose()` must return an assignment whose `var_ids` are pairwise distinct
// and each index a live (lb < ub) dimension (design/MODULE_REFACTOR.md §4.5)
// -- GreedyCompositionPolicy satisfies this by construction (fill_binary/
// fill_integer/fill_continuous each visit distinct vids and partition by
// VarKind), and GraphDriver asserts it in debug builds.
template<typename T>
class CompositionPolicy
{
public:
  explicit CompositionPolicy(FanOutSpec fan_out,
                             SearchCalibration calibration = {})
      : fan_out_(fan_out)
      , calibration_(calibration)
  {
    // 0 means "unset": default to the full search cap.
    if (calibration_.max_cycle_size == 0) {
      calibration_.max_cycle_size = kMaxSlots;
    }
    if (calibration_.max_cycle_size > kMaxSlots) {
      throw InvalidConfiguration(
          "max_cycle_size (" + std::to_string(calibration_.max_cycle_size)
          + ") exceeds kMaxSlots (" + std::to_string(kMaxSlots) + ")");
    }
  }

  virtual ~CompositionPolicy() = default;

  // The fan-outs every consumer of this policy's SlotAssignments must decode
  // against. Not virtual: there is one answer per policy instance, fixed at
  // construction.
  const FanOutSpec& fan_out() const { return fan_out_; }

  const SearchCalibration& calibration() const { return calibration_; }

  // Slots this policy will ever fill.
  std::size_t max_cycle_size() const { return calibration_.max_cycle_size; }

  virtual SlotAssignment choose(
      std::span<const cu::interval<T>> box,
      std::span<const dag::VarKind> var_kinds) const = 0;

private:
  FanOutSpec fan_out_;
  SearchCalibration calibration_;
};

// Greedy, stateless composition policy: fills slots from unresolved (i.e.
// non-degenerate, lb < ub) variables, binaries first, then integers, then
// continuous.
template<typename T>
class GreedyCompositionPolicy : public CompositionPolicy<T>
{
public:
  using CompositionPolicy<T>::CompositionPolicy;
  using CompositionPolicy<T>::fan_out;
  using CompositionPolicy<T>::max_cycle_size;

  SlotAssignment choose(std::span<const cu::interval<T>> box,
                        std::span<const dag::VarKind> var_kinds) const override
  {
    assert(box.size() == var_kinds.size());
    if (var_kinds.empty()) {
      // Nothing to assign var_ids[s] = 0 to -- 0 isn't a valid variable
      // index on a zero-variable problem (TEST_EXTENSION.md, the
      // "every var_id < var_kinds.size()" invariant).
      throw ShapeMismatch(
          "CompositionPolicy::choose called with an empty var_kinds span; "
          "there are no " "variables to assign to any slot");
    }

    SlotAssignment out {};

    fill_binary(box, var_kinds, out);
    fill_integer(box, var_kinds, out);
    fill_continuous(box, var_kinds, out);

    return out;
  }

  // Number of integers in [ceil(b.lb), floor(b.ub)]. b need not itself be
  // lattice-aligned -- reachable directly from an IntegerPartition child,
  // whose boundaries reuse the continuous linear-width formula
  // (TEST_EXTENSION.md) -- so ceil/floor do the snapping here rather than
  // assuming the caller already aligned them. Returns 0, rather than
  // underflowing (UB, in fact: casting a negative double to size_t) a huge
  // size_t, when the sub-box contains no integer at all (ceil(b.lb) >
  // floor(b.ub)): this can't be enumerated or partitioned further and is
  // deterministically empty. Public (rather than an implementation-detail
  // private helper) because it's the exact locus TEST_EXTENSION.md calls out
  // by name.
  static std::size_t integer_domain_size(const cu::interval<T>& b)
  {
    T const lo = std::ceil(b.lb);
    T const hi = std::floor(b.ub);
    if (hi < lo) {
      return 0;
    }
    return static_cast<std::size_t>(hi - lo) + 1;
  }

private:
  static bool unresolved(const cu::interval<T>& b) { return b.ub > b.lb; }

  void append(SlotAssignment& out, SlotKind kind, std::size_t vid) const
  {
    out.composition.kinds.push_back(kind);
    out.var_ids.push_back(vid);
  }

  bool at_cap(const SlotAssignment& out) const
  {
    return out.var_ids.size() >= max_cycle_size();
  }

  // All three fill helpers stop at the policy's cap: a composition is
  // exactly its live slots, so a box with more live variables than the cap
  // simply leaves the rest unassigned this iteration.
  void fill_binary(std::span<const cu::interval<T>> box,
                   std::span<const dag::VarKind> var_kinds,
                   SlotAssignment& out) const
  {
    for (std::size_t vid = 0; vid < var_kinds.size() && !at_cap(out); ++vid) {
      if (var_kinds[vid] == dag::VarKind::Binary && unresolved(box[vid])) {
        append(out, SlotKind::BinaryEnumerate, vid);
      }
    }
  }

  // Non-static, unlike its binary/continuous siblings: it's the one that
  // consults enumerate_cap, which now lives on the policy instance.
  void fill_integer(std::span<const cu::interval<T>> box,
                    std::span<const dag::VarKind> var_kinds,
                    SlotAssignment& out) const
  {
    std::vector<std::size_t> candidates;
    for (std::size_t vid = 0; vid < var_kinds.size(); ++vid) {
      if (var_kinds[vid] == dag::VarKind::Integer && unresolved(box[vid])) {
        candidates.push_back(vid);
      }
    }
    // Smallest remaining domain first: most likely to become enumerable (or
    // already is), so this makes the fastest progress toward fully
    // resolving a dimension.
    std::sort(
        candidates.begin(),
        candidates.end(),
        [&](std::size_t a, std::size_t b)
        { return integer_domain_size(box[a]) < integer_domain_size(box[b]); });

    std::size_t const enumerate_cap = fan_out().enumerate_cap();
    for (std::size_t vid : candidates) {
      if (at_cap(out)) {
        break;
      }
      SlotKind const kind = integer_domain_size(box[vid]) <= enumerate_cap
          ? SlotKind::IntegerEnumerate
          : SlotKind::IntegerPartition;
      append(out, kind, vid);
    }
  }

  void fill_continuous(std::span<const cu::interval<T>> box,
                       std::span<const dag::VarKind> var_kinds,
                       SlotAssignment& out) const
  {
    std::vector<std::size_t> candidates;
    for (std::size_t vid = 0; vid < var_kinds.size(); ++vid) {
      if (var_kinds[vid] == dag::VarKind::Continuous && unresolved(box[vid])) {
        candidates.push_back(vid);
      }
    }
    // Widest first: classic largest-uncertainty-first bisection heuristic.
    std::sort(candidates.begin(),
              candidates.end(),
              [&](std::size_t a, std::size_t b)
              { return (box[a].ub - box[a].lb) > (box[b].ub - box[b].lb); });

    for (std::size_t vid : candidates) {
      if (at_cap(out)) {
        break;
      }
      append(out, SlotKind::Continuous, vid);
    }
  }
};

// True iff every var_id in `assignment` is unique and indexes a live
// (lb < ub) dimension of `box` -- CompositionPolicy's contract (§4.5), a
// debug-only check (see GraphDriver::solve()). GreedyCompositionPolicy
// satisfies it by construction; nothing asserted it before this.
template<typename T>
bool assignment_is_distinct_and_live(const SlotAssignment& assignment,
                                     std::span<const cu::interval<T>> box)
{
  for (std::size_t j = 0; j < assignment.var_ids.size(); ++j) {
    std::size_t const vid = assignment.var_ids[j];
    if (!(box[vid].lb < box[vid].ub)) {
      return false;
    }
    for (std::size_t k = j + 1; k < assignment.var_ids.size(); ++k) {
      if (assignment.var_ids[k] == vid) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace cuminlp
