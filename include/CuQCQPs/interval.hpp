#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include <cuinterval/cuinterval.h>

namespace cuqcqps::interval
{

// ```interval``` from ```interval.h``` is used to represent a box in one
// dimension whenever a multi-dimensional box is required, store them as a
// vector.

// Forward declared so CompressedInterval::materialise can take a reference
// to the history it is looked up in.
template<typename T>
struct IntervalHistory;  // will store all previous expanded interval<T>
                         // instances.

/**
 * @brief Compressed format for an interval, used to store all intervals pending
 * in the list and all intervals that have been explored.
 *
 * @tparam T
 */
template<typename T>
struct CompressedInterval
{
  std::size_t sidx;  // index within parent interval
  std::size_t pidx;  // index of parent interval in history
  std::size_t cycle_start;  // start of block of dimensions to partition
  T lb;  // sound lower bound over this interval

  bool operator<(const CompressedInterval<T>& other) const
  {
    if (lb != other.lb) {
      return lb < other.lb;
    }
    return pidx < other.pidx;
  }

  bool operator==(const CompressedInterval& other) const
  {
    return lb == other.lb && pidx == other.pidx;
  }

  // Reconstructs the explicit bounds for this interval, looking up its
  // parent in `history` if it has one and narrowing by this interval's own
  // sidx/cycle_start. `out.bounds` must already be sized to the problem's
  // dimensionality. `cycle_size`/`partition_num` mirror the CycleSize and
  // PartitionNum used to build the opt::CycleContext that produced this
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
    // block of dimensions, mirroring opt::make_cycle_context.
    std::vector<std::size_t> part(cycle_size);
    std::size_t idx = sidx;
    for (std::size_t j = 0; j < cycle_size; ++j) {
      part[j] = idx % partition_num;
      idx /= partition_num;
    }

    // Narrow each dimension, mirroring opt::get_bounds.
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

// Min-Heap of Intervals for the pending list
// highest priority is least lb, then least pidx
template<typename T>
class IntervalPQueue
{
private:
  std::vector<CompressedInterval<T>> elems;
  std::size_t num_elems = 0;

  void grow() { elems.resize(elems.empty() ? 1 : elems.size() * 2); }

  void sift_up(std::size_t idx)
  {
    CompressedInterval<T> e = elems[idx];
    while (!is_root(idx) && e < elems[parent_idx(idx)]) {
      elems[idx] = elems[parent_idx(idx)];
      idx = parent_idx(idx);
    }
    elems[idx] = e;
  }

  void sift_down(std::size_t idx)
  {
    CompressedInterval<T> e = elems[idx];
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

  void enqueue(const CompressedInterval<T>& new_elem)
  {
    if (num_elems == elems.size()) {
      grow();
    }

    elems[num_elems] = new_elem;
    sift_up(num_elems);
    ++num_elems;
  }

  CompressedInterval<T> dequeue()
  {
    assert(num_elems > 0 && "dequeue() called on empty IntervalPQueue");

    CompressedInterval<T> res = elems[0];
    --num_elems;
    if (num_elems > 0) {
      elems[0] = elems[num_elems];
      sift_down(0);
    }
    return res;
  }

  bool empty() const { return num_elems == 0; }

  std::size_t size() const { return num_elems; }

  const CompressedInterval<T>& peek() const { return elems[0]; }

  // Number of pending regions not yet proven suboptimal against gub, i.e.
  // regions that could still contain the global optimum.
  std::size_t count_viable(T gub) const { return count_viable_impl(0, gub); }
};

// History of exploration just represented as a vector
template<typename T>
struct IntervalHistory
{
  std::vector<std::vector<cu::interval<T>>> intervals;

  explicit IntervalHistory(std::size_t initial_size = 0)
      : intervals(initial_size)
  {
  }

  // Appends an interval to the history, returning its index so it can be
  // used as the pidx of CompressedInterval entries derived from it.
  std::size_t enqueue(std::vector<cu::interval<T>> new_interval)
  {
    std::size_t idx = intervals.size();
    intervals.push_back(std::move(new_interval));
    return idx;
  }
};

}  // namespace cuqcqps::interval
