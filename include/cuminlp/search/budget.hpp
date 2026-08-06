#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>

// The host-side counterpart to backend::graph's device `budget_bytes`: a cap
// on the two unbounded host structures a search grows (the pending frontier
// and the interval history), the arithmetic that decides how far to compact
// the frontier when it is reached, and the accounting that keeps discarding
// regions *sound*.
namespace cuminlp::search
{

/// Why a solve stopped. Reported verbatim on the driver's `Stop reason:` line,
/// which is a machine-readable interface (tools/minlp_status.py) as much as a
/// human one -- these spellings are the contract.
enum class StopReason
{
  Converged,  ///< the bracket closed to tolerance
  Exhausted,  ///< the frontier emptied
  IterationLimit,  ///< ran out of iterations with regions still pending
  HostMemory,  ///< the host budget was reached with the frontier compacted
  AllocationFailure,  ///< a host allocation threw std::bad_alloc
  DeviceMemory,  ///< a mid-run graph build threw ResourceExhausted
};

inline const char* to_string(StopReason reason)
{
  switch (reason) {
    case StopReason::Converged:
      return "converged";
    case StopReason::Exhausted:
      return "exhausted";
    case StopReason::IterationLimit:
      return "iteration-limit";
    case StopReason::HostMemory:
      return "host-memory";
    case StopReason::AllocationFailure:
      return "allocation-failure";
    case StopReason::DeviceMemory:
      return "device-memory";
  }
  return "unknown";
}

/**
 * @brief What to do when the host budget is reached.
 *
 * The two are separated because only one of them changes what the search
 * explores. `StopAtBudget` discards nothing: the run ends through the ordinary
 * epilogue with its frontier intact, so every bound it reports is the bound it
 * would have reported anyway, and the only difference from an unbudgeted run
 * is that this one ends with a result instead of an OOM kill. `Compact` trades
 * regions for a longer run, and a region traded away is one the search will
 * not look in -- sound but a different search. That
 * makes it opt-in: `--bounded-frontier`.
 */
enum class FrontierPolicy
{
  StopAtBudget,  ///< end the run; discard nothing
  Compact,  ///< halve the frontier and keep going
};

/**
 * @brief MemAvailable from /proc/meminfo, in bytes, or 0 if it cannot be read.
 *
 * MemAvailable rather than MemFree: the kernel's own estimate of what a new
 * allocation can actually get hold of, which is the question being asked.
 * Everything here is Linux-specific and fails soft -- a 0 return means "not
 * measured", and the caller decides what to do about that.
 */
inline std::size_t available_host_bytes()
{
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  while (meminfo >> key) {
    if (key != "MemAvailable:") {
      meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    unsigned long long value = 0;
    std::string unit;
    if (!(meminfo >> value)) {
      return 0;
    }
    meminfo >> unit;
    return static_cast<std::size_t>(value) * (unit == "kB" ? 1024u : 1u);
  }
  return 0;
}

/// Denominator of the fraction of MemAvailable a measured budget takes. Half,
/// conservatively: the frontier's growth is bursty (one fan-out of children
/// per iteration, and grow() doubles), so a budget checked once per iteration
/// wants headroom for the overshoot between two checks.
inline constexpr std::size_t kMeasuredHostBudgetDivisor = 2;

/**
 * @brief The byte cap a solve actually runs under.
 *
 * Mirrors the device-side convention (GraphReplay::build): a nonzero request
 * is used as given, and 0 means "measured at solve() start". A measurement
 * that fails also resolves to 0, which every caller reads as "unbudgeted" --
 * refusing to run because /proc could not be read would be worse than the
 * unbounded behaviour that preceded this budget existing.
 */
inline std::size_t resolve_host_budget(std::size_t requested)
{
  if (requested != 0) {
    return requested;
  }
  return available_host_bytes() / kMeasuredHostBudgetDivisor;
}

/**
 * @brief Whether the host structures have reached `budget_bytes`.
 *
 * `>=`, not `>`, and that is load-bearing rather than a rounding preference:
 * compact_keep_count sizes the frontier so a compacted vector's capacity lands
 * exactly on the room the budget leaves after one doubling, so a strict `>`
 * would wave that doubling through and only notice after the *next* one -- at
 * twice the budget.
 *
 * A budget of 0 means unbudgeted (resolve_host_budget), and nothing is ever
 * over it.
 */
inline bool over_host_budget(std::size_t live_bytes, std::size_t budget_bytes)
{
  return budget_bytes != 0 && live_bytes >= budget_bytes;
}

/**
 * @brief How many frontier nodes to keep when compacting under `budget`.
 *
 * A *quarter* of the nodes the budget can hold once the live history has taken
 * its share -- not the half this returned until measurement showed what half
 * costs. `frontier_bytes` charges the backing vector's capacity (§3.3) and
 * `compact` leaves a capacity of `2 * n_keep` behind, so keeping half the room
 * handed back a frontier whose capacity *alone* was the whole room: the next
 * iteration's history entry put the total back over the budget, and compaction
 * ran again. Every iteration, evicting one fan-out each time, for an
 * O(frontier) nth_element and a full backing-vector copy apiece. Measured on
 * sporttournament10 at a 256 MiB cap: 0.6 s per post-budget iteration against
 * 0.009 s for the search itself, and 475 compactions over 600 iterations where
 * the interval below predicts one per ~50 (measured after the fix: 10).
 *
 * A quarter restores the amortisation the batching argument rests on:
 *
 *     after a compaction    capacity 2 * n_keep = room / 2   strictly under
 *     n_keep insertions on  capacity 4 * n_keep = room       at the budget
 *
 * so node count oscillates in [n_keep, 2 * n_keep], capacity in
 * [room / 2, room], and compaction runs once per n_keep insertions with the
 * peak allocation still exactly the budget and no more. The `>=` in
 * over_host_budget is the other half of this: the trip has to fire on a
 * capacity that lands *on* the room, because the next doubling is past it.
 *
 * Floored at 1: a history that has eaten the whole budget leaves no room to
 * divide, and the caller's job then is to stop, not to empty the queue. It
 * finds out by asking again after the compaction (which will still be over
 * budget) rather than by reading a 0 back from here.
 */
inline std::size_t compact_keep_count(std::size_t budget_bytes,
                                      std::size_t history_bytes,
                                      std::size_t node_bytes)
{
  std::size_t const room = budget_bytes - std::min(budget_bytes, history_bytes);
  return std::max<std::size_t>(1, room / (4 * node_bytes));
}

/**
 * @brief What a run discarded from its frontier, and the floor that keeps
 *        discarding sound.
 *
 * Evicting a pending region may lose the true optimum, so every claim the
 * epilogue makes has to be gated on what was evicted. One scalar suffices:
 * the least lb over all *viable* evictions. Below it, nothing was thrown away.
 *
 * A region whose lb already exceeds the incumbent is not folded in, and that
 * is not an approximation: GUB_ is monotone non-increasing, so a region above
 * it now can never hold a point better than the final incumbent. Discarding it
 * is the same pruning the child loop does, applied late -- which is what makes
 * eviction *lossless* for as long as anything dominated is still in the queue,
 * and since eviction takes worst-lb first, the dominated go first.
 */
struct DropAccounting
{
  /// Least lb over viable evictions; +inf while there have been none, which
  /// reads as "this run never discarded anything that could have mattered".
  double lb_min = std::numeric_limits<double>::infinity();
  std::size_t viable = 0;
  std::size_t dominated = 0;  ///< reporting only

  void fold(double lb, double gub)
  {
    if (lb <= gub) {
      lb_min = std::min(lb_min, lb);
      ++viable;
    } else {
      ++dominated;
    }
  }

  bool lost_nothing() const { return viable == 0; }
};

/// Is `bound` within tolerance of the incumbent? The driver's own convergence
/// test, relative to |gub| and floored at 1.0; shared so the epilogue below
/// applies exactly the same test to the dropped floor that the loop applies to
/// the frontier.
inline bool gap_closed(double gub, double bound, double tolerance)
{
  return gub - bound <= tolerance * std::max(1.0, std::fabs(gub));
}

}  // namespace cuminlp::search
