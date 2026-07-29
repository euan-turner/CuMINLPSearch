#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <cuinterval/cuinterval.h>

#include "cuminlp/composition_policy.hpp"

namespace cuminlp::search
{

// ```interval``` from ```interval.h``` is used to represent a box in one
// dimension whenever a multi-dimensional box is required, store them as a
// vector.

template<typename T>
struct IntervalHistory
{
  std::vector<std::vector<cu::interval<T>>> intervals;

  explicit IntervalHistory(std::size_t initial_size = 0)
      : intervals(initial_size)
  {
  }

  // Appends an interval to the history, returning its index so it can be
  // used as the pidx of CompressedInterval/CompositionInterval entries
  // derived from it.
  std::size_t enqueue(std::vector<cu::interval<T>> new_interval)
  {
    std::size_t idx = intervals.size();
    intervals.push_back(std::move(new_interval));
    return idx;
  }
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
  std::size_t depth; // tree depth of interval
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
 * @tparam T
 * @tparam CycleSize
 * @tparam PartitionNum
 * @tparam EnumerateCap
 */
template<typename T, std::size_t CycleSize, std::size_t PartitionNum,
        std::size_t EnumerateCap = PartitionNum>
struct CompositionInterval
{
  std::size_t sidx;  // index within parent interval
  std::size_t pidx;  // index of parent interval in history
  std::size_t depth; // tree depth of interval
  T lb;  // sound lower bound over this interval

  bool operator<(const CompositionInterval& other) const
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
  // CompositionPolicy's class comment), which is then decoded exactly as
  // partition::get_slot_bounds does on device.
  void materialise(const IntervalHistory<T>& history,
                   std::vector<cu::interval<T>>& out,
                   const cuminlp::CompositionPolicy<T, CycleSize>& policy,
                   std::span<const dag::VarKind> var_kinds) const
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
    out = parent;

    cuminlp::SlotAssignment<CycleSize> assignment = policy.choose(parent, var_kinds);

    // Decode sidx into a per-slot partition/enumeration index, mirroring
    // partition::make_slot_context, then narrow each cycled dimension,
    // mirroring partition::get_slot_bounds.
    std::size_t idx = sidx;
    for (std::size_t j = 0; j < CycleSize; ++j) {
      SlotKind kind = assignment.composition[j];
      std::size_t fan_out = cuminlp::slot_fan_out<PartitionNum, EnumerateCap>(kind);
      std::size_t part = idx % fan_out;
      idx /= fan_out;

      std::size_t dim = assignment.var_ids[j];
      const cu::interval<T>& pb = parent[dim];
      switch (kind) {
        case SlotKind::IntegerEnumerate:
        case SlotKind::BinaryEnumerate: {
          T value = pb.lb + static_cast<T>(part);
          if (value > pb.ub) value = pb.ub;
          out[dim].lb = value;
          out[dim].ub = value;
          break;
        }
        case SlotKind::Continuous:
        case SlotKind::IntegerBisect: {
          T width = (pb.ub - pb.lb) / static_cast<T>(fan_out);
          out[dim].lb = pb.lb + width * static_cast<T>(part);
          out[dim].ub = pb.lb + width * static_cast<T>(part + 1);
          break;
        }
      }
    }
  }
};

// Min-Heap of Intervals for the pending list
// highest priority is least lb, then least pidx
// Node defaults to FixedRosenbrockDriver's CompressedInterval<T>; GraphDriver
// instantiates this with CompositionInterval<T, CycleSize, PartitionNum>
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
};

}  // namespace cuminlp::search
