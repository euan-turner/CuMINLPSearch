#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>

// The host-side counterpart to graph_replay.cuh's device `budget_bytes`: a cap
// on the two unbounded host structures a search grows (the pending frontier
// and the interval history), the arithmetic that decides how far to compact
// the frontier when it is reached, and the accounting that keeps discarding
// regions *sound*. See design/BOUNDED_FRONTIER.md.
//
// Deliberately device-free, for the same reason search_sizing.hpp is: every
// decision here is arithmetic on counts and bounds, and none of it needs a GPU
// to test. graph_driver.cuh is where they are wired to an actual search.
namespace cuminlp
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
 * not look in -- sound (§4's floor sees to that) but a different search. That
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
 * @brief How many frontier nodes to keep when compacting under `budget`.
 *
 * Half the nodes the budget can hold once the live history has taken its
 * share, so the frontier oscillates in [n_keep, 2 * n_keep] and compaction
 * runs about once per n_keep insertions -- the amortisation the batching
 * argument rests on.
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
  return std::max<std::size_t>(1, room / (2 * node_bytes));
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

/// The epilogue's three answers.
struct FinalBounds
{
  double glb;  ///< the lower bound to report
  bool proven_optimal;  ///< the incumbent is optimal, and this run proved it
  bool infeasible;  ///< the frontier emptied and nothing viable was dropped
};

/**
 * @brief Resolve the final bracket and what may be claimed about it.
 *
 * @param glb            the lower bound as the loop left it (returned as-is on
 *                       the infeasible branch, which asserts nothing about it)
 * @param frontier_min   least pending lb; ignored when `pending_empty`
 * @param viable         pending regions not yet dominated by `gub`
 *
 * Every claim is gated on `dropped`:
 *
 * - the reported GLB gains the dropped floor as one more argument to the same
 *   clamp, because reporting a lower bound above a region nothing excluded is
 *   not a weak bound but a false one;
 * - proven optimality gains `gap_closed(dropped.lb_min)`, trivially true when
 *   nothing viable was dropped, so a memory-capped run whose evictions were
 *   all dominated still finishes proven optimal;
 * - infeasibility may only be claimed when nothing viable was dropped at all.
 *   Otherwise the frontier is empty because we emptied it, and the dropped
 *   regions are unexplored places rather than excluded ones.
 */
inline FinalBounds finalise_bounds(double glb,
                                   double gub,
                                   double tolerance,
                                   bool found_incumbent,
                                   bool converged,
                                   bool pending_empty,
                                   double frontier_min,
                                   std::size_t viable,
                                   const DropAccounting& dropped)
{
  FinalBounds out {glb, false, false};

  // Three ways to have proved it, not two. `converged` is the loop noticing
  // that the least pending lb already exceeds gub; an empty frontier is the
  // same thing having consumed the queue. But hitting the iteration limit with
  // every remaining region already dominated proves just as much, and used to
  // be reported as an unfinished run.
  out.proven_optimal = found_incumbent
      && (converged || pending_empty || viable == 0)
      && gap_closed(gub, dropped.lb_min, tolerance);

  if (out.proven_optimal) {
    out.glb = gub;
  } else if (pending_empty && dropped.lost_nothing()) {
    out.infeasible = true;  // glb left as the loop had it
  } else {
    // Clamped, never merely assigned. A region is enqueued only when its
    // interval lb is <= the gub *at the time*, so a later improvement to the
    // incumbent can leave the whole frontier above it; the sound global lower
    // bound is then the incumbent, not the frontier's minimum.
    double const frontier =
        pending_empty ? std::numeric_limits<double>::max() : frontier_min;
    out.glb = std::min(frontier, dropped.lb_min);
    if (found_incumbent) {
      out.glb = std::min(out.glb, gub);
    }
  }
  return out;
}

}  // namespace cuminlp
