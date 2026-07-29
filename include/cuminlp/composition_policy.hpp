#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/dag.hpp"

namespace cuminlp {

// Which operation a slot in a Composition performs on its assigned variable
// this iteration. This is what fixes a replayed CUDA
// graph's fan-out/shape, so each distinct sequence of SlotKinds corresponds
// to its own pre-built graph.
enum class SlotKind { Continuous, IntegerBisect, IntegerEnumerate, BinaryEnumerate };

template<std::size_t CycleSize>
using Composition = std::array<SlotKind, CycleSize>;

// Which variable fills each slot of `composition`, for one search-tree node.
template<std::size_t CycleSize>
struct SlotAssignment {
  Composition<CycleSize> composition;
  std::array<std::size_t, CycleSize> var_ids;
};

// Number of children a slot of the given kind produces. Binary is always
// exactly 2 (its domain is always size 2 until resolved); Continuous/
// IntegerBisect share PartitionNum (the bisection width); IntegerEnumerate
// uses its own EnumerateCap, independent of PartitionNum, so raising the
// enumerate threshold doesn't force wider bisection elsewhere. EnumerateCap
// defaults to PartitionNum, so anything that doesn't care about the
// distinction (composition_fan_out's other callers, e.g.) doesn't need to
// mention it.
template<std::size_t PartitionNum, std::size_t EnumerateCap = PartitionNum>
constexpr std::size_t slot_fan_out(SlotKind kind)
{
  if (kind == SlotKind::BinaryEnumerate) return 2;
  if (kind == SlotKind::IntegerEnumerate) return EnumerateCap;
  return PartitionNum;
}

template<std::size_t PartitionNum, std::size_t EnumerateCap, std::size_t CycleSize>
constexpr std::size_t composition_fan_out(const Composition<CycleSize>& composition)
{
  std::size_t total = 1;
  for (SlotKind kind : composition) {
    total *= slot_fan_out<PartitionNum, EnumerateCap>(kind);
  }
  return total;
}

// True iff every slot enumerates (IntegerEnumerate/BinaryEnumerate) rather
// than bisecting or ranging over a live Continuous variable. Combined with a
// node's live-variable count being <= CycleSize (checked separately, since
// that's a property of a box, not of a Composition in isolation), this is
// exactly the condition under which every child this Composition produces
// is a fully-resolved point -- see GraphDriver's use of ExactGraphReplay.
template<std::size_t CycleSize>
constexpr bool is_fully_enumerable(const Composition<CycleSize>& composition)
{
  for (SlotKind kind : composition) {
    if (kind != SlotKind::IntegerEnumerate && kind != SlotKind::BinaryEnumerate) return false;
  }
  return true;
}

// Decides, for a search-tree node's current box, which CycleSize variables to
// act on next and how. Must be a pure function of currently-observable state
// (the box + the problem's variable kinds).
template<typename T, std::size_t CycleSize>
class CompositionPolicy {
public:
  virtual ~CompositionPolicy() = default;

  virtual SlotAssignment<CycleSize> choose(std::span<const cu::interval<T>> box,
                                            std::span<const dag::VarKind> var_kinds) const = 0;

  // Every Composition `choose` could return, for any box with these
  // var_kinds.
  virtual std::vector<Composition<CycleSize>> possible_compositions(
      std::span<const dag::VarKind> var_kinds) const = 0;
};

// Greedy, stateless composition policy: fills slots from unresolved (i.e.
// non-degenerate, lb < ub) variables, binaries first, then integers, then
// continuous.
template<typename T, std::size_t CycleSize, std::size_t PartitionNum,
        std::size_t EnumerateCap = PartitionNum>
class GreedyCompositionPolicy : public CompositionPolicy<T, CycleSize> {
public:
  SlotAssignment<CycleSize> choose(std::span<const cu::interval<T>> box,
                                    std::span<const dag::VarKind> var_kinds) const override
  {
    assert(box.size() == var_kinds.size());

    SlotAssignment<CycleSize> out {};
    std::size_t filled = 0;

    filled = fill_binary(box, var_kinds, out, filled);
    filled = fill_integer(box, var_kinds, out, filled);
    filled = fill_continuous(box, var_kinds, out, filled);

    // Every variable is already resolved (can happen right before a node is
    // fathomed): pad remaining slots by repeating the last variable
    // assigned. Re-bisecting or re-enumerating a fixed point is a no-op, so
    // this is safe, just wasted work.
    for (std::size_t s = filled; s < CycleSize; ++s) {
      out.composition[s] = SlotKind::Continuous;
      out.var_ids[s] = filled > 0 ? out.var_ids[filled - 1] : 0;
    }

    return out;
  }

  // `choose` always lays out slots as a run of BinaryEnumerate, then a run of
  // IntegerEnumerate, then a run of IntegerBisect, then Continuous for the
  // rest: fill_integer visits candidates in ascending domain-size order and
  // decides Enumerate vs Bisect by comparing each candidate's own domain size
  // against EnumerateCap, so once one candidate exceeds EnumerateCap, every
  // later (larger-or-equal) one does too -- the Enumerate/Bisect split within
  // the integer run can only happen at one point, not in an arbitrary
  // per-slot pattern. So every reachable composition is fixed by 3 counts:
  // how many leading slots are binary (k), how many of the following
  // integer slots enumerate (e) vs bisect (m - e), out of m total integer
  // slots. k and m are bounded not just by CycleSize but by how many
  // binary/integer variables the problem actually has -- fill_binary/
  // fill_integer can never fill more slots than there are live candidates of
  // that kind, at any node, ever.
  std::vector<Composition<CycleSize>> possible_compositions(
      std::span<const dag::VarKind> var_kinds) const override
  {
    std::size_t num_binary = 0;
    std::size_t num_integer = 0;
    for (dag::VarKind kind : var_kinds) {
      if (kind == dag::VarKind::Binary) ++num_binary;
      if (kind == dag::VarKind::Integer) ++num_integer;
    }
    std::size_t const max_k = std::min(CycleSize, num_binary);

    std::vector<Composition<CycleSize>> out;
    for (std::size_t k = 0; k <= max_k; ++k) {
      std::size_t const max_m = std::min(CycleSize - k, num_integer);
      for (std::size_t m = 0; m <= max_m; ++m) {
        for (std::size_t e = 0; e <= m; ++e) {
          Composition<CycleSize> c {};
          std::size_t s = 0;
          for (std::size_t i = 0; i < k; ++i) c[s++] = SlotKind::BinaryEnumerate;
          for (std::size_t i = 0; i < e; ++i) c[s++] = SlotKind::IntegerEnumerate;
          for (std::size_t i = 0; i < m - e; ++i) c[s++] = SlotKind::IntegerBisect;
          for (; s < CycleSize; ++s) c[s] = SlotKind::Continuous;
          out.push_back(c);
        }
      }
    }
    return out;
  }

private:
  static bool unresolved(const cu::interval<T>& b) { return b.ub > b.lb; }

  // Assumes bounds are already integer-valued, as required for
  // VarKind::Integer/Binary -- not yet enforced by a bound-tightening pass.
  static std::size_t integer_domain_size(const cu::interval<T>& b)
  {
    return static_cast<std::size_t>(std::floor(b.ub) - std::ceil(b.lb)) + 1;
  }

  static std::size_t fill_binary(std::span<const cu::interval<T>> box,
                                 std::span<const dag::VarKind> var_kinds,
                                 SlotAssignment<CycleSize>& out,
                                 std::size_t filled)
  {
    for (std::size_t vid = 0; vid < var_kinds.size() && filled < CycleSize; ++vid) {
      if (var_kinds[vid] == dag::VarKind::Binary && unresolved(box[vid])) {
        out.composition[filled] = SlotKind::BinaryEnumerate;
        out.var_ids[filled] = vid;
        ++filled;
      }
    }
    return filled;
  }

  static std::size_t fill_integer(std::span<const cu::interval<T>> box,
                                  std::span<const dag::VarKind> var_kinds,
                                  SlotAssignment<CycleSize>& out,
                                  std::size_t filled)
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
    std::sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) {
      return integer_domain_size(box[a]) < integer_domain_size(box[b]);
    });

    for (std::size_t vid : candidates) {
      if (filled >= CycleSize) break;
      out.composition[filled] = integer_domain_size(box[vid]) <= EnumerateCap
          ? SlotKind::IntegerEnumerate
          : SlotKind::IntegerBisect;
      out.var_ids[filled] = vid;
      ++filled;
    }
    return filled;
  }

  static std::size_t fill_continuous(std::span<const cu::interval<T>> box,
                                     std::span<const dag::VarKind> var_kinds,
                                     SlotAssignment<CycleSize>& out,
                                     std::size_t filled)
  {
    std::vector<std::size_t> candidates;
    for (std::size_t vid = 0; vid < var_kinds.size(); ++vid) {
      if (var_kinds[vid] == dag::VarKind::Continuous && unresolved(box[vid])) {
        candidates.push_back(vid);
      }
    }
    // Widest first: classic largest-uncertainty-first bisection heuristic.
    std::sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) {
      return (box[a].ub - box[a].lb) > (box[b].ub - box[b].lb);
    });

    for (std::size_t vid : candidates) {
      if (filled >= CycleSize) break;
      out.composition[filled] = SlotKind::Continuous;
      out.var_ids[filled] = vid;
      ++filled;
    }
    return filled;
  }
};

}  // namespace cuminlp
