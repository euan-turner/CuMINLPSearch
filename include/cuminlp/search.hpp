#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>
#include <limits>
#include <span>
#include <utility>
#include <vector>

// Only field access (.lb/.ub) on cu::interval<T> below, never interval
// arithmetic -- interval.h (a plain struct definition) is enough, and unlike
// cuinterval.h (which pulls in operations.cuh's unconditional __device__
// arithmetic overloads) it's compilable by a plain host compiler, which is
// what keeps this header's search::CompositionInterval::materialise
// host-testable without a CUDA toolchain (see TEST_EXTENSION.md).
#include <cuinterval/interval.h>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/slot_decode.hpp"

namespace cuminlp::search
{

// ```interval``` from ```interval.h``` is used to represent a box in one
// dimension whenever a multi-dimensional box is required, store them as a
// vector.

// A history entry is only ever read back as the direct parent of a dequeued
// node (CompositionInterval::materialise's `history.intervals[pidx]` lookup),
// so an entry with no pending children is garbage the moment its last child
// leaves the queue. Refcounting says exactly when that is: an entry's count is
// the construction reference enqueue() hands back, plus one per pending node
// naming it as pidx. See design/BOUNDED_FRONTIER.md §6.
//
// The counting is opt-in. A caller that never calls release() (
// FixedRosenbrockDriver, source/fixed_rosenbrock_driver.cu) never drops a
// count to zero, so nothing is ever freed and nothing is ever reused -- which
// is today's behaviour exactly, indices included.
template<typename T>
struct IntervalHistory
{
  std::vector<std::vector<cu::interval<T>>> intervals;

  explicit IntervalHistory(std::size_t initial_size = 0)
      : intervals(initial_size)
      , refs_(initial_size, 0)
      , live_bytes_(initial_size * entry_bytes(0))
  {
  }

  // Appends an interval to the history, returning the slot holding it with one
  // reference -- the "construction reference" -- held by the caller. The slot
  // index is what CompressedInterval/CompositionInterval entries derived from
  // this box carry as their pidx.
  //
  // A slot freed by release() is reused here rather than growing the vector,
  // so an index is only unique among *live* entries: pidx is no longer a
  // monotone recency stamp, which costs operator<'s third tie-break (see
  // there) and nothing else.
  std::size_t enqueue(std::vector<cu::interval<T>> new_interval)
  {
    std::size_t const bytes = entry_bytes(new_interval.size());
    if (!free_slots_.empty()) {
      std::size_t const idx = free_slots_.back();
      free_slots_.pop_back();
      intervals[idx] = std::move(new_interval);
      refs_[idx] = 1;
      live_bytes_ += bytes;
      return idx;
    }
    std::size_t const idx = intervals.size();
    intervals.push_back(std::move(new_interval));
    refs_.push_back(1);
    live_bytes_ += bytes;
    return idx;
  }

  // One more pending node names `idx` as its pidx.
  void add_ref(std::size_t idx)
  {
    if (idx == kRootSlot) {
      return;  // see release()
    }
    assert(idx < refs_.size() && "add_ref() on a slot that was never enqueued");
    assert(refs_[idx] > 0 && "add_ref() on a freed slot");
    ++refs_[idx];
  }

  // One fewer holder of `idx`. At zero references the box's memory is returned
  // and the slot joins the free list.
  //
  // Slot 0 is the root sentinel -- materialise() answers pidx == 0 from
  // root_box without ever looking the slot up -- so releasing it is a no-op and
  // it never enters the free list. That keeps index 0 meaning "the root" for
  // every node, which slot reuse would otherwise quietly break.
  void release(std::size_t idx)
  {
    if (idx == kRootSlot) {
      return;
    }
    assert(idx < refs_.size() && "release() on a slot that was never enqueued");
    assert(refs_[idx] > 0 && "release() on an already-freed slot");
    if (--refs_[idx] > 0) {
      return;
    }
    live_bytes_ -= entry_bytes(intervals[idx].size());
    // Not clear(): that keeps the capacity, and the whole point is to hand the
    // memory back.
    std::vector<cu::interval<T>>().swap(intervals[idx]);
    free_slots_.push_back(idx);
  }

  // Host bytes the live entries hold, for the driver's memory budget
  // (design/BOUNDED_FRONTIER.md §3.3). Maintained incrementally rather than
  // summed on demand, since it is read once per iteration.
  std::size_t live_bytes() const { return live_bytes_; }

  // Entries currently holding a box. Bounded by the number of distinct parents
  // among the pending nodes once the driver releases what it dequeues.
  std::size_t live_count() const
  {
    return intervals.size() - free_slots_.size();
  }

  std::size_t ref_count(std::size_t idx) const { return refs_[idx]; }

private:
  static constexpr std::size_t kRootSlot = 0;

  // The box itself plus a fixed charge for the std::vector header holding it:
  // an entry costs more than its elements, and at 200 variables the header is
  // noise while at 2 it is not.
  static constexpr std::size_t entry_bytes(std::size_t n_vars)
  {
    return n_vars * sizeof(cu::interval<T>)
        + sizeof(std::vector<cu::interval<T>>);
  }

  std::vector<std::size_t> refs_;  // parallel to intervals
  std::vector<std::size_t> free_slots_;
  std::size_t live_bytes_ = 0;
};

/**
 * @brief Compressed format for an interval, used to store all intervals pending
 * in the list and all intervals that have been explored. This is
 * FixedRosenbrockDriver's own node type, cycling a fixed, contiguous
 * cycle_start block uniformly -- see CompositionInterval below for
 * GraphDriver's composition/policy-aware equivalent.
 *
 * @tparam T
 */
template<typename T>
struct CompressedInterval
{
  std::size_t sidx;  // index within parent interval
  std::size_t pidx;  // index of parent interval in history
  std::size_t depth;  // tree depth of interval
  std::size_t cycle_start;  // start of block of dimensions to partition
  T lb;  // sound lower bound over this interval

  bool operator<(const CompressedInterval<T>& other) const
  {
    // Lower is higher priority to explore
    // 1. by least lower bound
    // 2. by deepest in the tree
    // 3. by most recent parent
    if (lb != other.lb) {
      return lb < other.lb;
    }
    if (depth != other.depth) {
      return depth > other.depth;
    }
    return pidx > other.pidx;
  }

  bool operator==(const CompressedInterval& other) const
  {
    return lb == other.lb && pidx == other.pidx;
  }

  // Reconstructs the explicit bounds for this interval, looking up its
  // parent in `history` if it has one and narrowing by this interval's own
  // sidx/cycle_start. `out.bounds` must already be sized to the problem's
  // dimensionality. `cycle_size`/`partition_num` mirror the CycleSize and
  // PartitionNum used to build the partition::CycleContext that produced this
  // interval on device.
  void materialise(const IntervalHistory<T>& history,
                   std::vector<cu::interval<T>>& out,
                   std::size_t cycle_size,
                   std::size_t partition_num) const
  {
    if (pidx == 0) {
      // first interval - full (unbounded) domain
      for (auto& b : out) {
        b.lb = std::numeric_limits<T>::lowest();
        b.ub = std::numeric_limits<T>::max();
      }
      return;
    }

    const std::vector<cu::interval<T>>& parent = history.intervals[pidx];

    // Decode sidx into a per-dimension partition index within the cycled
    // block of dimensions, mirroring partition::make_cycle_context.
    std::vector<std::size_t> part(cycle_size);
    std::size_t idx = sidx;
    for (std::size_t j = 0; j < cycle_size; ++j) {
      part[j] = idx % partition_num;
      idx /= partition_num;
    }

    // Narrow each dimension, mirroring partition::get_bounds.
    for (std::size_t dim = 0; dim < out.size(); ++dim) {
      const cu::interval<T>& pb = parent[dim];
      if (dim >= cycle_start && dim < cycle_start + cycle_size) {
        std::size_t i = dim - cycle_start;
        T width = (pb.ub - pb.lb) / static_cast<T>(partition_num);
        out[dim].lb = pb.lb + width * static_cast<T>(part[i]);
        out[dim].ub = pb.lb + width * static_cast<T>(part[i] + 1);
      } else {
        out[dim] = pb;
      }
    }
  }
};

/**
 * @brief GraphDriver's node type: like CompressedInterval, but for a
 * CompositionPolicy-driven search instead of a fixed contiguous cycle_start
 * block. No cycle-position bookkeeping is stored here: CompositionPolicy is a
 * pure function of a node's box + variable kinds (see composition_policy.hpp),
 * so `materialise` recovers the SlotAssignment that produced this interval by
 * just re-invoking the policy on the reconstructed parent box, rather than by
 * tracking which slots/variables were used at each node.
 *
 * The fan-outs sidx is encoded against are read off `policy` in materialise()
 * rather than carried as template parameters, so they cannot disagree with
 * the ones the policy itself used (see CompositionPolicy::fan_out).
 *
 * @tparam T
 * @tparam Capacity
 */
template<typename T, std::size_t Capacity>
struct CompositionInterval
{
  std::size_t sidx;  // index within parent interval
  std::size_t pidx;  // index of parent interval in history
  std::size_t depth;  // tree depth of interval
  T lb;  // sound lower bound over this interval

  // Slots the policy filled when this node's sidx was encoded. Purely a
  // tripwire: materialise() re-derives the assignment anyway, so this exists
  // to catch the case where it re-derives a *different* one. See the
  // purity contract on CompositionPolicy -- a policy that consulted
  // mutable state would decode sidx against the wrong radix vector and
  // produce a silently wrong box rather than any kind of error. Two bytes
  // of the same information, compared, turns that into a thrown exception
  // at the exact point of violation.
  std::size_t slot_count = 0;

  bool operator<(const CompositionInterval& other) const
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

  bool operator==(const CompositionInterval& other) const
  {
    return lb == other.lb && pidx == other.pidx;
  }

  // Reconstructs the explicit bounds for this interval, looking up its
  // parent in `history` if it has one and narrowing by this interval's own
  // sidx. `out` must already be sized to the problem's dimensionality.
  // `policy`/`var_kinds` must be the same ones used to produce this interval
  // on device, so re-invoking `policy.choose()` on the reconstructed parent
  // box deterministically recovers the same SlotAssignment (see
  // CompositionPolicy's class comment), which is then decoded via the same
  // cuminlp::decode::slot_bounds partition::get_slot_bounds calls on device
  // (see TEST_EXTENSION.md) -- host/device agreement is structural, not
  // asserted by two hand-written implementations.
  //
  // `root_box` is the true root domain (Problem::box_bounds) for pidx == 0:
  // there is no parent to look up at the root, so it can't be reconstructed
  // from `history` the way every other node's box can. Passing it explicitly
  // keeps materialise() usable as the single decode path for every node,
  // including the root, rather than callers special-casing pidx == 0
  // themselves (see GraphDriver::solve()).
  void materialise(const IntervalHistory<T>& history,
                   std::vector<cu::interval<T>>& out,
                   const cuminlp::CompositionPolicy<T, Capacity>& policy,
                   std::span<const dag::VarKind> var_kinds,
                   std::span<const cu::interval<T>> root_box) const
  {
    if (pidx == 0) {
      out.assign(root_box.begin(), root_box.end());
      return;
    }

    const std::vector<cu::interval<T>>& parent = history.intervals[pidx];
    out = parent;

    cuminlp::SlotAssignment<Capacity> assignment =
        policy.choose(parent, var_kinds);

    // The purity tripwire (see slot_count above). If the policy did not
    // return the same assignment it returned when this node was enqueued,
    // every decode below is against the wrong radix vector and the box we
    // would hand back is wrong but entirely plausible-looking. Fail here
    // instead.
    if (assignment.composition.count != slot_count) {
      throw cuminlp::InvalidConfiguration(
          "CompositionPolicy::choose is not a pure function of (box, "
          "var_kinds): re-invoking it on this node's parent returned "
          + std::to_string(assignment.composition.count) + " slots, but "
          + std::to_string(slot_count)
          + " were used to encode the node's sidx. A policy must not depend "
            "on state that changes during a solve");
    }

    // Decode sidx into a per-slot partition/enumeration index, mirroring
    // partition::make_slot_context, then narrow each cycled dimension via
    // the same decode partition::get_slot_bounds uses on device.
    //
    // Live slots only. The padding tail must be skipped, not merely decoded
    // to a no-op: `choose` fills a Padding slot's var_id by *repeating the
    // last variable it assigned*, and slot_bounds() for Padding writes
    // `out[dim] = parent[dim]`. Walking the tail therefore resets that
    // variable's bounds to the parent's, silently undoing the narrowing an
    // earlier real slot applied to the same dimension. The device is not
    // affected: get_slot_bounds returns on its first var_ids match, so the
    // real slot always wins there -- which is exactly what made this a
    // host/device divergence rather than an obvious failure. Skipping the
    // tail restores agreement.
    //
    // Padding contributes fan-out 1, so stopping early leaves the radix
    // decode of sidx unchanged (`idx % 1 == 0`, `idx /= 1`).
    std::size_t idx = sidx;
    for (std::size_t j = 0; j < assignment.composition.count; ++j) {
      SlotKind kind = assignment.composition[j];
      std::size_t fan_out = cuminlp::slot_fan_out(kind, policy.fan_out());
      std::size_t part = idx % fan_out;
      idx /= fan_out;

      std::size_t dim = assignment.var_ids[j];
      cuminlp::decode::slot_bounds<T>(
          kind, parent[dim], part, fan_out, out[dim]);
    }
  }
};

// Min-Heap of Intervals for the pending list
// highest priority is least lb, then least pidx
// Node defaults to FixedRosenbrockDriver's CompressedInterval<T>; GraphDriver
// instantiates this with CompositionInterval<T, CycleSize>
// instead -- the heap logic itself only needs Node's operator</.lb, so it's
// shared between both node shapes rather than duplicated.
template<typename T, typename Node = CompressedInterval<T>>
class IntervalPQueue
{
private:
  std::vector<Node> elems;
  std::size_t num_elems = 0;

  void grow() { elems.resize(elems.empty() ? 1 : elems.size() * 2); }

  void sift_up(std::size_t idx)
  {
    Node e = elems[idx];
    while (!is_root(idx) && e < elems[parent_idx(idx)]) {
      elems[idx] = elems[parent_idx(idx)];
      idx = parent_idx(idx);
    }
    elems[idx] = e;
  }

  void sift_down(std::size_t idx)
  {
    Node e = elems[idx];
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

  void enqueue(const Node& new_elem)
  {
    if (num_elems == elems.size()) {
      grow();
    }

    elems[num_elems] = new_elem;
    sift_up(num_elems);
    ++num_elems;
  }

  Node dequeue()
  {
    assert(num_elems > 0 && "dequeue() called on empty IntervalPQueue");

    Node res = elems[0];
    --num_elems;
    if (num_elems > 0) {
      elems[0] = elems[num_elems];
      sift_down(0);
    }
    return res;
  }

  bool empty() const { return num_elems == 0; }

  std::size_t size() const { return num_elems; }

  const Node& peek() const { return elems[0]; }

  // Number of pending regions not yet proven suboptimal against gub, i.e.
  // regions that could still contain the global optimum.
  std::size_t count_viable(T gub) const { return count_viable_impl(0, gub); }

  // Elements the backing vector has room for, and what that costs. Capacity
  // rather than size(), because capacity is what is allocated and what grow()
  // doubles -- the allocation that actually fails is the doubling one, so a
  // budget checked against size() would be checked against a number the
  // process is already past. See design/BOUNDED_FRONTIER.md §3.3.
  std::size_t capacity() const { return elems.capacity(); }

  std::size_t capacity_bytes() const { return elems.capacity() * sizeof(Node); }

  // Keep the `n_keep` best elements under Node::operator<, hand each of the
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
      on_evict(static_cast<const Node&>(elems[i]));
    }
    num_elems = n_keep;

    // Floyd's heapify over the survivors: O(n), against O(n log n) to reinsert
    // them one at a time.
    for (std::size_t i = num_elems / 2; i-- > 0;) {
      sift_down(i);
    }

    if (elems.size() > 2 * n_keep) {
      std::vector<Node> shrunk(2 * n_keep);
      std::copy(elems.begin(),
                elems.begin() + static_cast<std::ptrdiff_t>(num_elems),
                shrunk.begin());
      elems.swap(shrunk);
    }
  }
};

}  // namespace cuminlp::search
