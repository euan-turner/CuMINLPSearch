#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuinterval/interval.h>

namespace cuminlp::search
{

// A history entry is only ever read back as the direct parent of a dequeued
// node (Node::materialise's `history.intervals[pidx]` lookup),
// so an entry with no pending children is garbage the moment its last child
// leaves the queue. Refcounting says exactly when that is: an entry's count is
// the construction reference enqueue() hands back, plus one per pending node
// naming it as pidx. See design/BOUNDED_FRONTIER.md §6.
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
  // index is what Node entries derived from this box carry as
  // their pidx.
  //
  // A slot freed by release() is reused here rather than growing the vector,
  // so an index is only unique among *live* entries: pidx is no longer a
  // monotone recency stamp, which costs operator<'s third tie-break (see
  // there) and nothing else.
  std::size_t enqueue(std::vector<cu::interval<T>> new_interval)
  {
    std::size_t const bytes = entry_bytes(new_interval.size());
    std::size_t idx = 0;
    if (!free_slots_.empty()) {
      idx = free_slots_.back();
      free_slots_.pop_back();
      intervals[idx] = std::move(new_interval);
      refs_[idx] = 1;
    } else {
      idx = intervals.size();
      intervals.push_back(std::move(new_interval));
      refs_.push_back(1);
    }
    live_bytes_ += bytes;
    peak_live_ = std::max(peak_live_, live_count());
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
    ++freed_;
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

  // Slots that hit refcount zero and joined the free list -- an entry that
  // was reused still counts once per release, same as one that grew the
  // vector instead. For the telemetry block's "history slots freed" line.
  std::uint64_t freed_count() const { return freed_; }

  // The high-water mark of live_count() over this history's lifetime. For
  // the telemetry block's "peak live" line -- a slot count, not bytes; the
  // byte figure is already the host budget's business (live_bytes()).
  std::size_t peak_live() const { return peak_live_; }

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
  std::uint64_t freed_ = 0;
  std::size_t peak_live_ = 0;
};

}  // namespace cuminlp::search
