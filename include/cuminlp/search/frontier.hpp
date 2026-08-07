#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Only field access (.lb/.ub) on cu::interval<T> below, never interval
// arithmetic -- interval.h (a plain struct definition) is enough, and unlike
// cuinterval.h (which pulls in operations.cuh's unconditional __device__
// arithmetic overloads) it's compilable by a plain host compiler, which is
// what keeps this header's search::Node::materialise
// host-testable without a CUDA toolchain (see TEST_EXTENSION.md).
#include <cuinterval/interval.h>

#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/materialise.hpp"
#include "cuminlp/search/history.hpp"

namespace cuminlp::search
{

/**
 * @brief SearchDriver's node type. CompositionPolicy is a pure function of a
 * node's box + variable kinds (see policy/policy.hpp), so `materialise`
 * recovers the SlotAssignment that produced this interval by just
 * re-invoking the policy on the reconstructed parent box, rather than by
 * tracking which slots/variables were used at each node.
 *
 * The fan-outs sidx is encoded against come from re-invoking `policy.choose()`
 * on the reconstructed parent box, which returns them on the SlotAssignment
 * itself (SlotAssignment::widths) -- they cannot disagree with the ones the
 * policy used to encode sidx in the first place, because it is the same call.
 *
 * @tparam T
 */
template<typename T>
struct Node
{
  std::size_t sidx;  // index within parent interval
  std::size_t pidx;  // index of parent interval in history
  std::size_t depth;  // tree depth of interval
  T lb;  // sound lower bound over this interval

  // A hash of the SlotAssignment (kinds, var_ids, widths) the policy filled
  // when this node's sidx was encoded. Purely a tripwire: materialise()
  // re-derives the assignment anyway, so this exists to catch the case
  // where it re-derives a *different* one. See the purity contract on
  // CompositionPolicy -- a policy that consulted mutable state would decode
  // sidx against the wrong radix vector and produce a silently wrong box
  // rather than any kind of error.
  //
  // A slot *count* alone (the old tripwire) cannot see this once widths are
  // data-dependent (design/BUDGETED_PARTITION.md §9): two assignments with
  // the same slot count but different widths decode sidx into completely
  // different, entirely plausible-looking boxes. Hashing `widths` in too is
  // what turns that into a thrown exception at the exact point of violation.
  std::uint64_t assignment_hash = 0;

  bool operator<(const Node& other) const
  {
    // Lower is higher priority to explore
    // 1. by least lower bound
    // 2. by deepest in the tree
    // 3. by most recent parent -- or, once IntervalHistory starts reusing the
    //    slots it frees, merely by the larger slot index: deterministic, but
    //    no longer a recency order. It only ever separates nodes with equal lb
    //    *and* equal depth, and nothing below depends on which of those wins.
    //
    // The lb tie is spelled with two < rather than a != so that a host
    // translation unit built with -Werror=float-equal can instantiate this
    // (test/source/host_budget_test.cpp does). Same ordering either way.
    if (lb < other.lb) {
      return true;
    }
    if (other.lb < lb) {
      return false;
    }
    if (depth != other.depth) {
      return depth > other.depth;
    }
    return pidx > other.pidx;
  }

  bool operator==(const Node& other) const
  {
    return lb == other.lb && pidx == other.pidx;
  }

  // Reconstructs the explicit bounds for this interval, looking up its
  // parent in `history` if it has one and narrowing by this interval's own
  // sidx. `out` must already be sized to the problem's dimensionality.
  // `policy`/`var_kinds` must be the same ones used to produce this interval
  // on device, so re-invoking `policy.choose()` on the reconstructed parent
  // box deterministically recovers the same SlotAssignment (see
  // CompositionPolicy's class comment), which is then decoded via
  // region::materialise -- the same decode the device's apply_slots_kernel
  // calls (see TEST_EXTENSION.md) -- so host/device agreement is structural,
  // not asserted by two hand-written implementations.
  //
  // `root_box` is the true root domain (Problem::box_bounds) for pidx == 0:
  // there is no parent to look up at the root, so it can't be reconstructed
  // from `history` the way every other node's box can. Passing it explicitly
  // keeps materialise() usable as the single decode path for every node,
  // including the root, rather than callers special-casing pidx == 0
  // themselves (see SearchDriver::solve()).
  void materialise(const IntervalHistory<T>& history,
                   std::vector<cu::interval<T>>& out,
                   const policy::CompositionPolicy<T>& policy,
                   std::span<const model::VarKind> var_kinds,
                   std::span<const cu::interval<T>> root_box) const
  {
    if (pidx == 0) {
      out.assign(root_box.begin(), root_box.end());
      return;
    }

    const std::vector<cu::interval<T>>& parent = history.intervals[pidx];

    region::SlotAssignment assignment = policy.choose(parent, var_kinds);

    // The purity tripwire (see assignment_hash above). If the policy did not
    // return the same assignment it returned when this node was enqueued,
    // every decode below is against the wrong radix vector and the box we
    // would hand back is wrong but entirely plausible-looking. Fail here
    // instead.
    std::uint64_t const rehashed = region::assignment_hash(assignment);
    if (rehashed != assignment_hash) {
      throw cuminlp::InvalidConfiguration(
          "CompositionPolicy::choose is not a pure function of (box, "
          "var_kinds): re-invoking it on this node's parent returned an "
          "assignment whose hash does not match the one used to encode the "
          "node's sidx. A policy must not depend on state that changes "
          "during a solve");
    }

    region::materialise<T>(parent, sidx, assignment, out);
  }
};

// Min-Heap of pending search-tree nodes.
// highest priority is least lb, then least pidx. One node type since the
// legacy path retired (design/MODULE_REFACTOR.md §6.3), so the element type
// is Node<T> rather than a second template parameter; the heap logic itself
// only needs Node's operator</.lb.
template<typename T>
class IntervalPQueue
{
private:
  using Element = Node<T>;

  std::vector<Element> elems;
  std::size_t num_elems = 0;

  void grow() { elems.resize(elems.empty() ? 1 : elems.size() * 2); }

  void sift_up(std::size_t idx)
  {
    Element e = elems[idx];
    while (!is_root(idx) && e < elems[parent_idx(idx)]) {
      elems[idx] = elems[parent_idx(idx)];
      idx = parent_idx(idx);
    }
    elems[idx] = e;
  }

  void sift_down(std::size_t idx)
  {
    Element e = elems[idx];
    while (has_left_child(idx)) {
      std::size_t child = smallest_child_idx(idx);
      if (!(elems[child] < e)) {
        break;
      }
      elems[idx] = elems[child];
      idx = child;
    }
    elems[idx] = e;
  }

  bool is_root(std::size_t idx) const { return idx == 0; }

  bool has_left_child(std::size_t idx) const
  {
    return left_child_idx(idx) < num_elems;
  }

  bool has_right_child(std::size_t idx) const
  {
    return right_child_idx(idx) < num_elems;
  }

  // Precondition: has_left_child(idx)
  std::size_t smallest_child_idx(std::size_t idx) const
  {
    std::size_t l = left_child_idx(idx);
    std::size_t r = right_child_idx(idx);
    return (has_right_child(idx) && elems[r] < elems[l]) ? r : l;
  }

  static std::size_t parent_idx(std::size_t idx) { return (idx - 1) / 2; }

  static std::size_t left_child_idx(std::size_t idx) { return 2 * idx + 1; }

  static std::size_t right_child_idx(std::size_t idx) { return 2 * idx + 2; }

  // Staleness is downward-closed: if a node's lb already exceeds gub, the
  // min-heap invariant guarantees every descendant's lb does too, so the
  // whole subtree can be skipped without visiting it.
  std::size_t count_viable_impl(std::size_t idx, T gub) const
  {
    if (idx >= num_elems || elems[idx].lb > gub) {
      return 0;
    }
    return 1 + count_viable_impl(left_child_idx(idx), gub)
        + count_viable_impl(right_child_idx(idx), gub);
  }

public:
  explicit IntervalPQueue(std::size_t capacity)
      : elems(capacity)
  {
  }

  void enqueue(const Element& new_elem)
  {
    if (num_elems == elems.size()) {
      grow();
    }

    elems[num_elems] = new_elem;
    sift_up(num_elems);
    ++num_elems;
  }

  Element dequeue()
  {
    assert(num_elems > 0 && "dequeue() called on empty IntervalPQueue");

    Element res = elems[0];
    --num_elems;
    if (num_elems > 0) {
      elems[0] = elems[num_elems];
      sift_down(0);
    }
    return res;
  }

  bool empty() const { return num_elems == 0; }

  std::size_t size() const { return num_elems; }

  const Element& peek() const { return elems[0]; }

  // Number of pending regions not yet proven suboptimal against gub, i.e.
  // regions that could still contain the global optimum.
  std::size_t count_viable(T gub) const { return count_viable_impl(0, gub); }

  // Elements the backing vector has room for, and what that costs. Capacity
  // rather than size(), because capacity is what is allocated and what grow()
  // doubles -- the allocation that actually fails is the doubling one, so a
  // budget checked against size() would be checked against a number the
  // process is already past. See design/BOUNDED_FRONTIER.md §3.3.
  std::size_t capacity() const { return elems.capacity(); }

  std::size_t capacity_bytes() const
  {
    return elems.capacity() * sizeof(Element);
  }

  // Keep the `n_keep` best elements under Element::operator<, hand each of the
  // others to `on_evict`, restore the heap invariant over the survivors, and
  // shrink the backing vector to hold 2 * n_keep. No-op when size() <= n_keep.
  //
  // The eviction policy is not designed here, and that is the point: the
  // "worst" element under the comparator the search already trusts is the
  // highest lb, then the shallowest, then the arbitrary tie-break -- loose
  // regions with bad bounds, which is exactly what a bounded frontier wants to
  // shed first. Discarding regions is only sound because the caller folds what
  // it drops into a floor on the reported lower bound; that contract lives
  // with the driver (design/BOUNDED_FRONTIER.md §4), not here.
  //
  // Batched rather than per-insertion: nth_element partitions in O(n) and the
  // heap rebuild is O(n), so keeping the frontier in [n_keep, 2 * n_keep]
  // amortises to a handful of comparisons per insertion -- cheaper than the
  // heap insert it rides along with -- and every eviction decision is made
  // with the freshest GUB the caller has.
  //
  // The capacity shrink is required rather than cosmetic: capacity_bytes()
  // counts capacity, so a vector that never gave any back would sit above the
  // budget line forever and re-trigger compaction on every later iteration.
  //
  // It shrinks to 2 * n_keep rather than to n_keep so the survivors can absorb
  // n_keep more insertions before the vector doubles -- which is the whole of
  // the amortisation, and which puts a constraint on n_keep that only the
  // caller can honour: the budget has to hold this capacity *and* the doubling
  // to 4 * n_keep that ends the interval. See compact_keep_count.
  template<typename OnEvict>
  void compact(std::size_t n_keep, OnEvict&& on_evict)
  {
    assert(n_keep >= 1 && "compact() must keep at least one element");
    if (num_elems <= n_keep) {
      return;
    }

    // Partitions the best n_keep to the front; neither side ends up ordered,
    // which is why the heap has to be rebuilt below rather than merely
    // truncated.
    std::nth_element(elems.begin(),
                     elems.begin() + static_cast<std::ptrdiff_t>(n_keep),
                     elems.begin() + static_cast<std::ptrdiff_t>(num_elems));

    for (std::size_t i = n_keep; i < num_elems; ++i) {
      on_evict(static_cast<const Element&>(elems[i]));
    }
    num_elems = n_keep;

    // Floyd's heapify over the survivors: O(n), against O(n log n) to reinsert
    // them one at a time.
    for (std::size_t i = num_elems / 2; i-- > 0;) {
      sift_down(i);
    }

    if (elems.size() > 2 * n_keep) {
      std::vector<Element> shrunk(2 * n_keep);
      std::copy(elems.begin(),
                elems.begin() + static_cast<std::ptrdiff_t>(num_elems),
                shrunk.begin());
      elems.swap(shrunk);
    }
  }
};

}  // namespace cuminlp::search
