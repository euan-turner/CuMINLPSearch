#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/aggregate/selection.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/materialise.hpp"
#include "cuminlp/search/history.hpp"

// The aggregate search's pending frontier (design/AGGREGATE_BOUNDING.md §7.3).
//
// Host-only, like `search/frontier.hpp` and for the same reason: nothing here
// needs a device, and the ordering logic is where the subtle bugs live.
namespace cuminlp::aggregate
{

/**
 * @brief One pending child.
 *
 * `sidx` indexes the **branch** assignment only -- `[0, k)`, not `[0, N)`.
 * That is the whole of §5.1's benefit: the box is re-derived by re-invoking
 * `AggregatePolicy::branch` on the reconstructed parent and decoding `sidx`
 * against the result, so only that small assignment carries a purity
 * contract, and the refine that actually produced this node's bound is never
 * re-derived at all.
 */
template<typename T>
struct AggregateNode
{
  std::size_t sidx = 0;  ///< child index within the parent's branch, [0, k)
  std::size_t pidx = 0;  ///< parent box's slot in search::IntervalHistory
  std::size_t depth = 0;
  T lb = std::numeric_limits<T>::lowest();  ///< sound; the dual bound
  T hull_ub = std::numeric_limits<T>::max();  ///< NOT an incumbent (§4.5)

  /// Hash of the branch assignment this node's `sidx` was encoded against --
  /// the same tripwire `search::Node` carries, over a smaller assignment.
  std::uint64_t branch_hash = 0;

  /**
   * @brief Rebuild this node's box from its parent's.
   *
   * @throws InvalidConfiguration if re-invoking `policy.branch` on the parent
   *         returns a different assignment from the one `sidx` was encoded
   *         against. Without this the decode would be against the wrong radix
   *         and would produce a box that is wrong but entirely plausible.
   */
  template<typename Policy>
  void materialise(const search::IntervalHistory<T>& history,
                   std::vector<cu::interval<T>>& out,
                   const Policy& policy,
                   std::span<const model::VarKind> var_kinds,
                   std::span<const cu::interval<T>> root_box) const
  {
    if (pidx == 0) {
      out.assign(root_box.begin(), root_box.end());
      return;
    }
    const std::vector<cu::interval<T>>& parent = history.intervals[pidx];
    region::SlotAssignment const branch = policy.branch(parent, var_kinds);
    if (region::assignment_hash(branch) != branch_hash) {
      throw cuminlp::InvalidConfiguration(
          "AggregatePolicy::branch is not a pure function of (box, "
          "var_kinds): re-invoking it on this node's parent returned a "
          "different assignment from the one the node's sidx was encoded "
          "against");
    }
    region::materialise<T>(parent, sidx, branch, out);
  }
};

/**
 * @brief The pending frontier, indexed twice.
 *
 * A node store plus two heaps of references into it (§7.3):
 *
 * - **by key** -- what to explore next, under the configured `SelectionRule`.
 * - **by `lb`** -- the least pending lower bound, for `GLB`, the convergence
 *   test, and `finalise_bounds`.
 *
 * `search::IntervalPQueue` needs only one heap because its pop order *is*
 * `lb` order, so the popped node is the frontier's `lb`-minimum by
 * construction. `RejectIndex` breaks that identity, and the two questions
 * genuinely come apart: reordering exploration must not be able to change the
 * reported dual bound. Keeping them in separate indices makes that structural
 * rather than something the driver has to remember.
 *
 * Deletion is lazy. A popped node is marked dead in the store and its stale
 * reference in the other heap is discarded when it surfaces. References carry
 * their own key and tie-break fields rather than reading them back from the
 * store, so a slot recycled under a stale reference cannot silently change
 * that reference's ordering and corrupt the heap invariant.
 */
template<typename T>
class AggregateFrontier
{
public:
  explicit AggregateFrontier(
      SelectionRule rule = SelectionRule::LeastLowerBound)
      : rule_(rule)
  {
  }

  SelectionRule rule() const { return rule_; }

  bool empty() const { return live_ == 0; }

  std::size_t size() const { return live_; }

  /// `f_best` is the incumbent the key is computed against; `+inf` before
  /// there is one. Under `LeastLowerBound` it is unused.
  void enqueue(const AggregateNode<T>& node, double f_best)
  {
    assert(!(node.hull_ub < node.lb)
           && "AggregateNode: hull_ub below lb; the two reductions disagreed "
              "about which subregions were excluded");

    std::size_t slot = 0;
    if (!free_slots_.empty()) {
      slot = free_slots_.back();
      free_slots_.pop_back();
      // ++gen retires every reference still pointing at this slot from its
      // previous occupant, so recycling cannot make a stale reference alias
      // the new node.
      slots_[slot] = Slot {node, slots_[slot].gen + 1, true};
    } else {
      slot = slots_.size();
      slots_.push_back(Slot {node, 0, true});
    }
    ++live_;

    push(by_key_, make_ref(slot, key_of(node, f_best)));
    push(by_lb_, make_ref(slot, static_cast<double>(node.lb)));
  }

  /// The best node under the configured rule. Precondition: `!empty()`.
  AggregateNode<T> pop()
  {
    assert(live_ > 0 && "pop() on an empty AggregateFrontier");
    while (!by_key_.empty()) {
      Ref const ref = top(by_key_);
      pop_top(by_key_);
      if (!is_current(ref)) {
        continue;  // popped through the other index, or slot since recycled
      }
      AggregateNode<T> const out = slots_[ref.slot].node;
      slots_[ref.slot].alive = false;
      --live_;
      free_slots_.push_back(ref.slot);
      collect_garbage();
      return out;
    }
    assert(false && "live_ > 0 but the key heap held no live reference");
    return AggregateNode<T> {};
  }

  /**
   * @brief The least pending lower bound; `+inf` when empty.
   *
   * Monotone non-decreasing across a search **regardless of pop order** --
   * but only because the driver clamps each child's `lb` to its parent's
   * (§4.4). That clamp is not cosmetic: an aggregate over a re-partitioned
   * child genuinely can be weaker than the child's own bound, since the
   * grandchildren's cover is a *different* cover rather than a refinement of
   * it. Unclamped, this value would still be a sound lower bound but could
   * visibly fall during a run.
   *
   * Given the clamp, monotonicity is what lets the selection rule change
   * without touching the dual bound's validity.
   *
   * Not const: it discards stale references off the top of the `lb` heap as
   * it goes, which is where that heap's garbage is actually collected.
   */
  T min_lb()
  {
    while (!by_lb_.empty() && !is_current(top(by_lb_))) {
      pop_top(by_lb_);
    }
    if (by_lb_.empty()) {
      return std::numeric_limits<T>::max();
    }
    return slots_[top(by_lb_).slot].node.lb;
  }

  /**
   * @brief Re-key every live node against an improved incumbent (§7.4).
   *
   * A no-op unless the rule's key actually depends on the incumbent. When it
   * does, a decrease in `f_best` does not shift all keys uniformly -- a node
   * with a narrow objective range moves more than one with a wide range -- so
   * the relative order genuinely changes and the old heap is invalid, not
   * merely stale.
   *
   * `O(n)`: one pass to re-key, then Floyd heapify, the same operation
   * `IntervalPQueue::compact` already performs. Incumbent improvements are
   * bounded by the number of strict `GUB` decreases, so this is rare.
   */
  void on_incumbent(double f_best)
  {
    if (!rebuilds_on_incumbent(rule_)) {
      return;
    }
    rebuild(f_best);
  }

  /// Pending nodes not yet proven suboptimal against `gub`. A linear scan
  /// over live slots, not the `lb` heap's downward-closed subtree skip: that
  /// skip is sound only on an `lb`-ordered heap, and this runs once, at the
  /// end of a solve, over a frontier of at most a few hundred thousand nodes.
  std::size_t count_viable(T gub) const
  {
    std::size_t viable = 0;
    for (const Slot& s : slots_) {
      if (s.alive && !(s.node.lb > gub)) {
        ++viable;
      }
    }
    return viable;
  }

  /// Host bytes this frontier holds, for the driver's budget check. Counts
  /// capacity rather than size, and counts dead slots and stale references,
  /// because that is what is actually allocated.
  std::size_t capacity_bytes() const
  {
    return slots_.capacity() * sizeof(Slot)
        + (by_key_.capacity() + by_lb_.capacity()) * sizeof(Ref)
        + free_slots_.capacity() * sizeof(std::size_t);
  }

  /// Live nodes, in unspecified order. For telemetry and for tests that need
  /// to assert over the whole frontier without draining it.
  std::vector<AggregateNode<T>> snapshot() const
  {
    std::vector<AggregateNode<T>> out;
    out.reserve(live_);
    for (const Slot& s : slots_) {
      if (s.alive) {
        out.push_back(s.node);
      }
    }
    return out;
  }

private:
  struct Slot
  {
    AggregateNode<T> node;
    std::uint64_t gen = 0;  ///< bumped on every reuse; see enqueue()
    bool alive = false;
  };

  /// A heap entry. Carries its own ordering fields rather than reading them
  /// from `slots_`: a stale reference whose slot has been recycled would
  /// otherwise change its comparison value after being pushed, which can
  /// break the heap invariant and strand a live entry. `gen` is what makes a
  /// reference to a recycled slot recognisable as stale at all -- `alive`
  /// alone cannot distinguish "this node is still pending" from "this slot
  /// holds a different node now".
  struct Ref
  {
    double key = 0.0;
    std::size_t depth = 0;
    std::size_t pidx = 0;
    std::size_t slot = 0;
    std::uint64_t gen = 0;
  };

  Ref make_ref(std::size_t slot, double key) const
  {
    const Slot& s = slots_[slot];
    return Ref {key, s.node.depth, s.node.pidx, slot, s.gen};
  }

  /// True iff `ref` still names the node it was created for.
  bool is_current(const Ref& ref) const
  {
    const Slot& s = slots_[ref.slot];
    return s.alive && s.gen == ref.gen;
  }

  double key_of(const AggregateNode<T>& node, double f_best) const
  {
    return selection_key<T>(rule_, node.lb, node.hull_ub, f_best);
  }

  /// True iff `a` should be popped before `b`. The tie-break chain is
  /// `search::Node::operator<`'s exactly -- key, then deepest, then largest
  /// parent slot -- so that under `LeastLowerBound` (whose key is `lb`) this
  /// frontier explores in the same order the existing one does. The two `<`
  /// rather than a `!=` on the key keeps this compilable under
  /// -Werror=float-equal, same as there.
  static bool is_better(const Ref& a, const Ref& b)
  {
    if (a.key < b.key) {
      return true;
    }
    if (b.key < a.key) {
      return false;
    }
    if (a.depth != b.depth) {
      return a.depth > b.depth;
    }
    return a.pidx > b.pidx;
  }

  /// std::pop_heap extracts the element the comparator calls *greatest*, so
  /// the comparator is `is_better` reversed: "greatest" means "best".
  static bool heap_less(const Ref& a, const Ref& b) { return is_better(b, a); }

  static void push(std::vector<Ref>& heap, const Ref& ref)
  {
    heap.push_back(ref);
    std::push_heap(heap.begin(), heap.end(), heap_less);
  }

  static const Ref& top(const std::vector<Ref>& heap) { return heap.front(); }

  static void pop_top(std::vector<Ref>& heap)
  {
    std::pop_heap(heap.begin(), heap.end(), heap_less);
    heap.pop_back();
  }

  /// Rebuild both indices from the live slots, re-keying against `f_best`.
  /// Also the only thing that reclaims stale references in bulk.
  void rebuild(double f_best)
  {
    by_key_.clear();
    by_lb_.clear();
    by_key_.reserve(live_);
    by_lb_.reserve(live_);
    for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
      if (!slots_[slot].alive) {
        continue;
      }
      by_key_.push_back(make_ref(slot, key_of(slots_[slot].node, f_best)));
      by_lb_.push_back(
          make_ref(slot, static_cast<double>(slots_[slot].node.lb)));
    }
    std::make_heap(by_key_.begin(), by_key_.end(), heap_less);
    std::make_heap(by_lb_.begin(), by_lb_.end(), heap_less);
  }

  /// Stale references accumulate one per pop, in whichever index the pop did
  /// not come through. Left alone they would grow without bound on a long
  /// run, so they are swept once they outnumber the live ones -- `O(n)` at
  /// half-density, hence amortised `O(1)` per pop.
  ///
  /// Re-keying here would be wrong under `RejectIndex`: this runs at an
  /// arbitrary moment with no incumbent in hand, so it preserves the keys the
  /// references already carry rather than inventing new ones.
  void collect_garbage()
  {
    std::size_t const refs = by_key_.size() + by_lb_.size();
    if (refs <= 2 * live_ + 64) {
      return;
    }
    auto sweep = [this](std::vector<Ref>& heap)
    {
      std::vector<Ref> kept;
      kept.reserve(live_);
      for (const Ref& r : heap) {
        if (is_current(r)) {
          kept.push_back(r);
        }
      }
      std::make_heap(kept.begin(), kept.end(), heap_less);
      heap.swap(kept);
    };
    sweep(by_key_);
    sweep(by_lb_);
  }

  SelectionRule rule_;
  std::vector<Slot> slots_;
  std::vector<std::size_t> free_slots_;
  std::vector<Ref> by_key_;
  std::vector<Ref> by_lb_;
  std::size_t live_ = 0;
};

}  // namespace cuminlp::aggregate
