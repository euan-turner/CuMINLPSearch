#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

#include "cuminlp/search/budget.hpp"

namespace cuminlp::search
{

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

}  // namespace cuminlp::search
