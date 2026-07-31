#pragma once

#include <algorithm>
#include <array>
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
// this iteration. This is what fixes a replayed CUDA
// graph's fan-out/shape, so each distinct sequence of SlotKinds corresponds
// to its own pre-built graph. Padding marks a slot with no live variable left
// to assign (every variable already resolved, or fewer live variables than
// CycleSize): it always has fan-out 1 and counts as enumerable, so a box
// whose live variables all enumerate can still be fathomed even when they
// don't fill every slot (TEST_EXTENSION.md §4b).
// Underlying type pinned to uint8_t: SlotKind is stored per-slot in the
// device-side SlotContext register array, where the default int would cost
// 4 bytes per slot for five enumerators (see design/RUNTIME_SHAPE.md §4.1).
enum class SlotKind : std::uint8_t
{
  Continuous,
  IntegerBisect,
  IntegerEnumerate,
  BinaryEnumerate,
  Padding
};

inline const char* slot_kind_name(SlotKind kind)
{
  switch (kind) {
    case SlotKind::Continuous:       return "Continuous";
    case SlotKind::IntegerBisect:    return "IntegerBisect";
    case SlotKind::IntegerEnumerate: return "IntegerEnumerate";
    case SlotKind::BinaryEnumerate:  return "BinaryEnumerate";
    case SlotKind::Padding:          return "Padding";
  }
  return "?";
}

// What each slot does this iteration, over a compile-time *capacity* with a
// runtime *count*. `Capacity` is the fixed array bound the device-side
// SlotContext is sized by (see design/RUNTIME_SHAPE.md §4); `count` is how
// many of those slots a given node actually fills.
//
// Slots at or past `count` are always Padding, so `count` is derivable from
// `kinds` and carrying it changes no equality or fan-out result -- two
// Compositions with equal `kinds` necessarily have equal `count`, which is
// why GraphDriver's per-Composition graph cache needs no change. It is
// stored because the *device* wants it: looping `j < count` instead of
// `j < Capacity` skips the padding tail, which at capacity 64 with three
// live variables is 61 wasted iterations per thread per variable.
template<std::size_t Capacity>
struct Composition
{
  std::array<SlotKind, Capacity> kinds {};
  std::size_t count = 0;

  bool operator==(const Composition&) const = default;

  // Range-for and indexing over the whole capacity, padding included: every
  // existing traversal wants all Capacity slots (the padding tail is
  // meaningful to is_fully_enumerable and contributes fan-out 1).
  constexpr auto begin() const { return kinds.begin(); }
  constexpr auto end() const { return kinds.end(); }
  constexpr SlotKind operator[](std::size_t i) const { return kinds[i]; }
  constexpr SlotKind& operator[](std::size_t i) { return kinds[i]; }
  constexpr const SlotKind* data() const { return kinds.data(); }
  static constexpr std::size_t capacity() { return Capacity; }

  // Every slot to `kind`, with the `count` choose() would have paired with
  // it: Capacity for a live kind, 0 for an all-Padding composition.
  constexpr void fill(SlotKind kind)
  {
    kinds.fill(kind);
    count = kind == SlotKind::Padding ? 0 : Capacity;
  }
};

// Which variable fills each slot of `composition`, for one search-tree node.
template<std::size_t Capacity>
struct SlotAssignment
{
  Composition<Capacity> composition;
  std::array<std::size_t, Capacity> var_ids {};

  std::size_t count() const { return composition.count; }
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
// IntegerBisect share partition_num (the bisection width); IntegerEnumerate
// uses its own enumerate_cap, independent of partition_num, so raising the
// enumerate threshold doesn't force wider bisection elsewhere.
inline std::size_t slot_fan_out(SlotKind kind, const FanOutSpec& fan_out)
{
  if (kind == SlotKind::BinaryEnumerate) {
    return 2;
  }
  if (kind == SlotKind::IntegerEnumerate) {
    return fan_out.enumerate_cap();
  }
  if (kind == SlotKind::Padding) {
    return 1;
  }
  return fan_out.partition_num();
}

// Total children a Composition produces -- the product of its slots'
// fan-outs, i.e. the `n_regions` a GraphReplay built for it will launch.
//
// Saturates at SIZE_MAX rather than wrapping. With partition_num a
// compile-time constant this product was always small and hand-checked; a
// runtime value makes e.g. CycleSize 20 with partition_num 10 (10^20, past
// the 1.8e19 size_t ceiling) reachable from a command line, and a wrapped
// product would silently size buffers *too small* instead of failing.
// Callers size allocations off this, so saturation turns that into an
// obvious ResourceExhausted at GraphReplay::build() -- see estimate_bytes().
template<std::size_t CycleSize>
std::size_t composition_fan_out(const Composition<CycleSize>& composition,
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

// True iff every slot enumerates (IntegerEnumerate/BinaryEnumerate) or is
// unused Padding, rather than bisecting or ranging over a live Continuous
// variable. Combined with a node's live-variable count fitting in the slots
// the policy actually filled (checked separately, since that's a property of
// a box, not of a Composition in isolation), this is exactly the condition
// under which every child this Composition produces is a fully-resolved
// point -- see GraphDriver's use of ExactGraphReplay. Padding counts as
// enumerable (its fan-out is always 1, i.e. it contributes no children) so
// that a box whose live variables all enumerate is still fathomable even
// when they don't fill every slot -- see can_fathom_without_children below.
template<std::size_t Capacity>
constexpr bool is_fully_enumerable(const Composition<Capacity>& composition)
{
  for (SlotKind kind : composition) {
    if (kind != SlotKind::IntegerEnumerate && kind != SlotKind::BinaryEnumerate
        && kind != SlotKind::Padding)
    {
      return false;
    }
  }
  return true;
}

// A node may be fathomed without enqueueing children iff every live variable
// fits in the slots this iteration actually filled and the resulting
// Composition is fully enumerable (TEST_EXTENSION.md §4b). Pure function of a
// box's live-variable count and the Composition the policy chose for it, so
// it's testable independent of the surrounding search loop (see
// graph_driver.cuh).
//
// The bound is `composition.count`, not `Capacity`. These coincided while
// the two were the same number, but a policy capped below its compiled
// capacity (--max-cycle-size, or a rung rounded up from the cap) fills only
// `count` slots: testing against `Capacity` would then claim a box was fully
// covered when `Capacity - count` of its live variables had no slot at all,
// and fathom a subtree that still contained the optimum.
template<std::size_t Capacity>
constexpr bool can_fathom_without_children(
    std::size_t live_count, const Composition<Capacity>& composition)
{
  return live_count <= composition.count && is_fully_enumerable(composition);
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
// (box, var_kinds, calibration). See design/RUNTIME_SHAPE.md §5.
struct SearchCalibration
{
  // Upper bound on slots the policy may fill. Independent of, and never
  // greater than, the Capacity the graphs were built for.
  std::size_t max_cycle_size = 0;
  std::size_t free_device_bytes = 0;
  std::size_t multiprocessor_count = 0;
};

// Decides, for a search-tree node's current box, which variables to act on
// next and how many slots to use. Must be a pure function of the box, the
// problem's variable kinds, and the frozen SearchCalibration -- nothing that
// can change during a run (see SearchCalibration above).
//
// The policy also *owns* the FanOutSpec. This is deliberate: decoding a
// node's box (search::CompositionInterval::materialise) needs both the
// SlotAssignment and the fan-outs its sidx was encoded against, and reading
// both off one object makes them impossible to disagree. Previously the
// driver and the policy each carried their own EnumerateCap template
// argument with nothing checking they matched -- a mismatch silently decoded
// sidx against the wrong radix, and source/gams/solve.cu worked around it by
// constructing both from one set of template parameters at a single call
// site.
template<typename T, std::size_t Capacity>
class CompositionPolicy
{
public:
  explicit CompositionPolicy(FanOutSpec fan_out,
                             SearchCalibration calibration = {})
      : fan_out_(fan_out)
      , calibration_(calibration)
  {
    // 0 means "unset": default to the full compiled capacity.
    if (calibration_.max_cycle_size == 0) {
      calibration_.max_cycle_size = Capacity;
    }
    if (calibration_.max_cycle_size > Capacity) {
      throw InvalidConfiguration(
          "max_cycle_size (" + std::to_string(calibration_.max_cycle_size)
          + ") exceeds the capacity this policy was instantiated for ("
          + std::to_string(Capacity)
          + "); pick a larger ladder rung instead of raising the cap");
    }
  }

  virtual ~CompositionPolicy() = default;

  // The fan-outs every consumer of this policy's SlotAssignments must decode
  // against. Not virtual: there is one answer per policy instance, fixed at
  // construction.
  const FanOutSpec& fan_out() const { return fan_out_; }

  const SearchCalibration& calibration() const { return calibration_; }

  // Slots this policy will ever fill: its cap, never above the capacity.
  std::size_t max_cycle_size() const { return calibration_.max_cycle_size; }

  virtual SlotAssignment<Capacity> choose(
      std::span<const cu::interval<T>> box,
      std::span<const dag::VarKind> var_kinds) const = 0;

  // Every Composition `choose` could return, for any box with these
  // var_kinds.
  virtual std::vector<Composition<Capacity>> possible_compositions(
      std::span<const dag::VarKind> var_kinds) const = 0;

private:
  FanOutSpec fan_out_;
  SearchCalibration calibration_;
};

// Greedy, stateless composition policy: fills slots from unresolved (i.e.
// non-degenerate, lb < ub) variables, binaries first, then integers, then
// continuous.
template<typename T, std::size_t Capacity>
class GreedyCompositionPolicy : public CompositionPolicy<T, Capacity>
{
public:
  using CompositionPolicy<T, Capacity>::CompositionPolicy;
  using CompositionPolicy<T, Capacity>::fan_out;
  using CompositionPolicy<T, Capacity>::max_cycle_size;

  SlotAssignment<Capacity> choose(
      std::span<const cu::interval<T>> box,
      std::span<const dag::VarKind> var_kinds) const override
  {
    assert(box.size() == var_kinds.size());
    if (var_kinds.empty()) {
      // Nothing to assign var_ids[s] = 0 to -- 0 isn't a valid variable
      // index on a zero-variable problem (TEST_EXTENSION.md §3, the
      // "every var_id < var_kinds.size()" invariant).
      throw ShapeMismatch(
          "CompositionPolicy::choose called with an empty var_kinds span; "
          "there are no " "variables to assign to any slot");
    }

    SlotAssignment<Capacity> out {};
    std::size_t filled = 0;

    filled = fill_binary(box, var_kinds, out, filled);
    filled = fill_integer(box, var_kinds, out, filled);
    filled = fill_continuous(box, var_kinds, out, filled);

    // Every live variable is already assigned (or there are none left,
    // which can happen right before a node is fathomed): pad remaining
    // slots as Padding, fan-out 1, repeating the last variable assigned
    // (or variable 0 if there was none). A Padding slot contributes no
    // children, so this is safe and keeps a fully-resolved box whose live
    // variables fit in `filled` slots fully enumerable -- see
    // can_fathom_without_children.
    for (std::size_t s = filled; s < Capacity; ++s) {
      out.composition[s] = SlotKind::Padding;
      out.var_ids[s] = filled > 0 ? out.var_ids[filled - 1] : 0;
    }

    // The count varies per node: a box with three live binaries fills three
    // slots regardless of the capacity the graphs were built for. Slots past
    // it are Padding, so this is a device-side loop bound, not a change in
    // what the composition means (see Composition's comment).
    out.composition.count = filled;

    return out;
  }

  // `choose` always lays out slots as a run of BinaryEnumerate, then a run of
  // IntegerEnumerate, then a run of IntegerBisect, then Continuous, then
  // Padding for whatever's left: fill_integer visits candidates in ascending
  // domain-size order and decides Enumerate vs Bisect by comparing each
  // candidate's own domain size against enumerate_cap, so once one candidate
  // exceeds EnumerateCap, every later (larger-or-equal) one does too -- the
  // Enumerate/Bisect split within the integer run can only happen at one
  // point, not in an arbitrary per-slot pattern. So every reachable
  // composition is fixed by 4 counts: how many leading slots are binary (k),
  // how many of the following integer slots enumerate (e) vs bisect (m - e)
  // out of m total integer slots, and how many of the remaining slots are
  // genuinely continuous (c) vs unused Padding. k, m and c are bounded not
  // just by CycleSize but by how many binary/integer/continuous variables
  // the problem actually has -- fill_binary/fill_integer/fill_continuous can
  // never fill more slots than there are live candidates of that kind, at
  // any node, ever.
  std::vector<Composition<Capacity>> possible_compositions(
      std::span<const dag::VarKind> var_kinds) const override
  {
    std::size_t num_binary = 0;
    std::size_t num_integer = 0;
    std::size_t num_continuous = 0;
    for (dag::VarKind kind : var_kinds) {
      if (kind == dag::VarKind::Binary) {
        ++num_binary;
      }
      if (kind == dag::VarKind::Integer) {
        ++num_integer;
      }
      if (kind == dag::VarKind::Continuous) {
        ++num_continuous;
      }
    }
    // Bounded by the policy's cap, not the compiled capacity: a rung wider
    // than --max-cycle-size must not enumerate compositions choose() would
    // never return, or the driver would size its ladder rung off phantom
    // entries.
    std::size_t const slots = max_cycle_size();
    std::size_t const max_k = std::min(slots, num_binary);

    std::vector<Composition<Capacity>> out;
    for (std::size_t k = 0; k <= max_k; ++k) {
      std::size_t const max_m = std::min(slots - k, num_integer);
      for (std::size_t m = 0; m <= max_m; ++m) {
        for (std::size_t e = 0; e <= m; ++e) {
          std::size_t const remaining = slots - k - m;
          std::size_t const max_c = std::min(remaining, num_continuous);
          for (std::size_t c = 0; c <= max_c; ++c) {
            Composition<Capacity> comp {};
            std::size_t s = 0;
            for (std::size_t i = 0; i < k; ++i) {
              comp[s++] = SlotKind::BinaryEnumerate;
            }
            for (std::size_t i = 0; i < e; ++i) {
              comp[s++] = SlotKind::IntegerEnumerate;
            }
            for (std::size_t i = 0; i < m - e; ++i) {
              comp[s++] = SlotKind::IntegerBisect;
            }
            for (std::size_t i = 0; i < c; ++i) {
              comp[s++] = SlotKind::Continuous;
            }
            comp.count = s;
            for (; s < Capacity; ++s) {
              comp[s] = SlotKind::Padding;
            }
            out.push_back(comp);
          }
        }
      }
    }
    return out;
  }

  // Number of integers in [ceil(b.lb), floor(b.ub)]. b need not itself be
  // lattice-aligned -- reachable directly from an IntegerBisect child, whose
  // boundaries reuse the continuous linear-width formula (TEST_EXTENSION.md
  // §3b) -- so ceil/floor do the snapping here rather than assuming the
  // caller already aligned them. Returns 0, rather than underflowing (UB, in
  // fact: casting a negative double to size_t) a huge size_t, when the
  // sub-box contains no integer at all (ceil(b.lb) > floor(b.ub)): this
  // can't be enumerated or bisected further and is deterministically empty.
  // Public (rather than an implementation-detail private helper) because
  // it's the exact locus TEST_EXTENSION.md §3b calls out by name.
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

  // All three fill helpers stop at the policy's cap rather than at Capacity,
  // so a rung wider than --max-cycle-size leaves the surplus slots Padding.
  std::size_t fill_binary(std::span<const cu::interval<T>> box,
                          std::span<const dag::VarKind> var_kinds,
                          SlotAssignment<Capacity>& out,
                          std::size_t filled) const
  {
    for (std::size_t vid = 0; vid < var_kinds.size() && filled < max_cycle_size();
         ++vid)
    {
      if (var_kinds[vid] == dag::VarKind::Binary && unresolved(box[vid])) {
        out.composition[filled] = SlotKind::BinaryEnumerate;
        out.var_ids[filled] = vid;
        ++filled;
      }
    }
    return filled;
  }

  // Non-static, unlike its binary/continuous siblings: it's the one that
  // consults enumerate_cap, which now lives on the policy instance.
  std::size_t fill_integer(std::span<const cu::interval<T>> box,
                           std::span<const dag::VarKind> var_kinds,
                           SlotAssignment<Capacity>& out,
                           std::size_t filled) const
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
      if (filled >= max_cycle_size()) {
        break;
      }
      out.composition[filled] = integer_domain_size(box[vid]) <= enumerate_cap
          ? SlotKind::IntegerEnumerate
          : SlotKind::IntegerBisect;
      out.var_ids[filled] = vid;
      ++filled;
    }
    return filled;
  }

  std::size_t fill_continuous(std::span<const cu::interval<T>> box,
                              std::span<const dag::VarKind> var_kinds,
                              SlotAssignment<Capacity>& out,
                              std::size_t filled) const
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
      if (filled >= max_cycle_size()) {
        break;
      }
      out.composition[filled] = SlotKind::Continuous;
      out.var_ids[filled] = vid;
      ++filled;
    }
    return filled;
  }
};

}  // namespace cuminlp
