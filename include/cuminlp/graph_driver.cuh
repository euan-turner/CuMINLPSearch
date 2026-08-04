#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/cuda_utils.cuh"
#include "cuminlp/cuminlp.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/graph_replay.cuh"
#include "cuminlp/host_budget.hpp"
#include "cuminlp/search.hpp"

namespace cuminlp
{

// Driver for an arbitrary dag::Problem. Each iteration, the CompositionPolicy
// picks a SlotAssignment for the current node's box; the corresponding
// (point, interval) GraphReplay pair -- point for GUB candidates, interval
// for sound lower bounds / pruning -- is built lazily on first use and
// cached (find_graphs/find_exact_graphs), since possible_compositions() can
// be large and most entries may go unused for a given problem/Capacity.
//
// If a node's live dimensions all fit within Capacity and the chosen
// Composition is fully enumerable, an ExactGraphReplay evaluates it exactly
// in one shot and fathoms it directly instead of enqueueing children for
// interval pruning.
//
// CompositionPolicy is a runtime constructor argument (not a template
// parameter) so it can be chosen via CLI flag without recompiling. The
// fan-out widths (formerly the PartitionNum/EnumerateCap template
// parameters) come from that same policy via CompositionPolicy::fan_out, so
// the driver and the policy cannot disagree about them -- previously they
// were independent template arguments whose agreement nothing checked.
template<typename T, std::size_t Capacity>
class GraphDriver : public driver
{
public:
  // `sample_points` is how many points the point graph draws per subdomain
  // (formerly the SamplePoints template parameter); it does not reach the
  // interval or exact graphs, which evaluate one element per region.
  // `budget_bytes` caps device memory per built graph; 0 means "whatever is
  // free at the time each graph is built". See GraphReplay::build.
  //
  // `host_budget_bytes` caps the two unbounded *host* structures -- the
  // pending frontier and the live interval history -- taken together, with the
  // same convention: 0 means "measured at solve() start" (half of
  // /proc/meminfo's MemAvailable). Reaching it ends the run through the
  // ordinary epilogue with the bracket intact, instead of leaving the OOM
  // killer to throw the bracket away.
  //
  // `frontier_policy` decides what reaching it means, and defaults to the
  // option that changes nothing about the search: FrontierPolicy::Compact
  // halves the frontier and keeps going, which trades unexplored regions for
  // a longer run and is therefore opt-in. See design/BOUNDED_FRONTIER.md.
  explicit GraphDriver(
      std::shared_ptr<const CompositionPolicy<T, Capacity>> policy,
      uint32_t iter_limit = 1000000,
      double tolerance = 1e-6,
      std::size_t sample_points = 1,
      std::size_t budget_bytes = 0,
      std::size_t host_budget_bytes = 0,
      FrontierPolicy frontier_policy = FrontierPolicy::StopAtBudget)
      : driver(iter_limit, tolerance)
      , policy_(std::move(policy))
      , sample_points_(sample_points)
      , budget_bytes_(budget_bytes)
      , host_budget_bytes_(host_budget_bytes)
      , frontier_policy_(frontier_policy)
  {
    if (policy_ == nullptr) {
      throw cuminlp::InvalidConfiguration(
          "GraphDriver requires a non-null CompositionPolicy");
    }
  }

  // The sampled point that attained GUB_, indexed by variable. Empty until
  // solve() finds a feasible sample.
  std::span<const T> best_point() const { return best_point_; }

  auto solve(const dag::Problem<T>& problem) -> double
  {
    using search::CompositionInterval;
    using search::IntervalHistory;
    using search::IntervalPQueue;

    // Full double precision throughout: at this magnitude-vs-tolerance ratio
    // (e.g. objective ~1e3, tolerance ~1e-9) the default 6-sig-fig cout
    // format prints the same rounded value for boxes that are still ~1e-8
    // apart, which reads as "converged" long before gap_closed() agrees.
    std::streamsize const prev_precision = std::cout.precision(17);

    std::span<const dag::VarKind> const var_kinds = problem.var_kinds;
    FanOutSpec const& fan_out = policy_->fan_out();

    // One (point, interval) replay pair per Composition actually encountered,
    // built lazily and cached -- eagerly building every possible_compositions()
    // entry could waste enormous GPU memory on graphs nothing ever launches.
    //
    // `stamp` is what makes the cache *bounded*: see evict_lru below.
    struct CompositionGraphs
    {
      Composition<Capacity> composition;
      dag::PointGraphReplay<T, Capacity> point;
      dag::IntervalGraphReplay<T, Capacity> interval;
      std::size_t stamp = 0;  ///< use_clock at the last hit
    };

    // Same lazy/cached approach, one ExactGraphReplay per fully-enumerable
    // Composition actually encountered.
    struct ExactGraphs
    {
      Composition<Capacity> composition;
      dag::ExactGraphReplay<T, Capacity> exact;
      std::size_t stamp = 0;
    };

    std::vector<CompositionGraphs> graphs;
    std::vector<ExactGraphs> exact_graphs;
    std::size_t use_clock = 0;
    std::size_t evictions = 0;

    /*
     * Give one cached Composition's device memory back, least recently used
     * first. Returns false when there is nothing left to give.
     *
     * Without this the caches were unbounded, and that made
     * dag::auto_max_cycle_size unsound no matter how carefully it charged a
     * composition: it fits a cap to `composition_footprint_bytes`, which is
     * by definition what *one* Composition costs, while a search that reaches
     * n distinct Compositions held all n at once. On batch.gms the resolver
     * spent two thirds of free memory on a cap of 14 -- correctly, for one
     * composition -- and the third distinct Composition the search reached
     * then failed to build. No fraction fixes that: the policy admits over a
     * hundred Compositions on a problem with both binaries and continuous
     * variables, and dividing the budget by that would leave a cap of 1.
     *
     * Eviction is sound because these are pure caches: an entry is a function
     * of its Composition and the problem, `set_domain` supplies everything
     * that varies per node, and a rebuilt entry replays identically. The cost
     * is the rebuild, paid only when the device is actually full.
     */
    auto evict_lru = [&]() -> bool
    {
      std::size_t oldest = std::numeric_limits<std::size_t>::max();
      std::size_t idx = 0;
      bool from_exact = false;
      bool found = false;
      for (std::size_t i = 0; i < graphs.size(); ++i) {
        if (graphs[i].stamp < oldest) {
          oldest = graphs[i].stamp;
          idx = i;
          from_exact = false;
          found = true;
        }
      }
      for (std::size_t i = 0; i < exact_graphs.size(); ++i) {
        if (exact_graphs[i].stamp < oldest) {
          oldest = exact_graphs[i].stamp;
          idx = i;
          from_exact = true;
          found = true;
        }
      }
      if (!found) {
        return false;
      }
      if (from_exact) {
        exact_graphs.erase(exact_graphs.begin() + static_cast<std::ptrdiff_t>(idx));
      } else {
        graphs.erase(graphs.begin() + static_cast<std::ptrdiff_t>(idx));
      }
      if (evictions++ == 0) {
        std::cout << "Device graph cache full: evicting the least recently "
                     "used composition's graphs and rebuilding as the search "
                     "revisits them.\n";
      }
      return true;
    };

    // Retry a cache fill, evicting until it fits. Only when this driver is
    // sizing against whatever the device has free (budget_bytes_ == 0): under
    // an explicit per-graph budget, freeing memory cannot change the verdict,
    // so evicting the cache would cost the rebuilds and still rethrow.
    auto fill_cache = [&](auto&& build_entry)
    {
      for (;;) {
        try {
          build_entry();
          return;
        } catch (const cuminlp::ResourceExhausted&) {
          if (budget_bytes_ != 0 || !evict_lru()) {
            throw;
          }
        }
      }
    };

    auto find_graphs =
        [&](const Composition<Capacity>& composition) -> CompositionGraphs&
    {
      for (auto& g : graphs) {
        if (g.composition == composition) {
          g.stamp = ++use_clock;
          return g;
        }
      }
      fill_cache(
          [&]
          {
            graphs.push_back(CompositionGraphs {
                composition,
                dag::PointGraphReplay<T, Capacity>::build(
                    problem, composition, fan_out, budget_bytes_, sample_points_),
                // sample_points_ reaches the interval graph only so its
                // out-of-memory report can cost a suggested cap against the
                // point graph too; build() clamps its own allocation to one
                // element per region regardless (GraphReplay::build,
                // `is_point && !Exact`).
                dag::IntervalGraphReplay<T, Capacity>::build(
                    problem, composition, fan_out, budget_bytes_, sample_points_),
                ++use_clock,
            });
          });
      return graphs.back();
    };

    auto find_exact_graphs =
        [&](const Composition<Capacity>& composition) -> ExactGraphs*
    {
      if (!is_fully_enumerable(composition)) {
        return nullptr;
      }
      for (auto& g : exact_graphs) {
        if (g.composition == composition) {
          g.stamp = ++use_clock;
          return &g;
        }
      }
      fill_cache(
          [&]
          {
            exact_graphs.push_back(ExactGraphs {
                composition,
                // Likewise for the exact graph: reporting only, never
                // allocation.
                dag::ExactGraphReplay<T, Capacity>::build(
                    problem, composition, fan_out, budget_bytes_, sample_points_),
                ++use_clock,
            });
          });
      return &exact_graphs.back();
    };

    using Node = CompositionInterval<T, Capacity>;

    IntervalPQueue<T, Node> pending(1000);
    IntervalHistory<T> history;

    // Slot 0, the root sentinel: materialise() answers pidx == 0 from
    // root_box without ever looking it up, so this entry exists only to make
    // index 0 mean "the root". It is never released and never reused.
    std::vector<cu::interval<T>> origin {};
    history.enqueue(origin);

    // Half of MemAvailable unless a cap was asked for. Printed once, because a
    // log has to record what the run actually ran under -- the number is a
    // property of the machine at solve() start, not of the command line.
    std::size_t const host_budget = resolve_host_budget(host_budget_bytes_);
    bool const compacting = frontier_policy_ == FrontierPolicy::Compact;
    if (host_budget == 0) {
      std::cout << "Host memory budget: none (nothing requested, and "
                   "MemAvailable could not be read from /proc/meminfo)\n";
    } else {
      std::cout << "Host memory budget: " << host_budget << " bytes ("
                << (host_budget_bytes_ == 0 ? "measured" : "requested")
                << "), on reaching it: "
                << (compacting ? "compact the frontier and continue"
                               : "stop with the frontier intact")
                << '\n';
    }

    pending.enqueue(CompositionInterval<T, Capacity> {
        .sidx = 0,
        .pidx = 0,
        .depth = 0,
        .lb = GLB_,
        // The root has no parent to decode against, so materialise() returns
        // root_box directly and never reaches the tripwire.
        .slot_count = 0,
    });

    bool converged = false;
    std::size_t pruned_infeasible = 0;

    // What the frontier gave up, and the floor that keeps giving it up sound.
    // See design/BOUNDED_FRONTIER.md §4 and DropAccounting.
    DropAccounting dropped;

    // Set at every break out of the loop below; when nothing breaks, the loop
    // condition says why it ended and the reason is derived after it.
    StopReason stop_reason = StopReason::IterationLimit;
    bool stopped_early = false;

    // tolerance_ is now relative to |GUB_| (floored at 1.0), not absolute:
    // an absolute 1e-9 on an objective of magnitude ~1e3 asks for ~1e-12
    // relative precision, which a wide-fan-out policy can take thousands of
    // extra iterations to certify even after the bound has visually
    // "arrived" at the printed value. bound is the tightest lower bound
    // known for the remaining search (the least pending lb, or GLB_).
    auto const gap_closed = [this](double bound) {
      return cuminlp::gap_closed(GUB_, bound, tolerance_);
    };

    // The loop is wrapped so a failed allocation ends the run through the same
    // epilogue as a spent iteration budget: the bracket GLB_ <= min <= GUB_ is
    // valid at every iteration, and letting the exception out of solve() would
    // throw it away. This is the seatbelt, not the brake -- on Linux with
    // default overcommit the usual failure is a SIGKILL no C++ mechanism sees,
    // which is what the host budget below is for.
    try {
      while (iter_idx_ < iter_limit_ && !pending.empty() && !gap_closed(GLB_)) {
        ++iter_idx_;

        CompositionInterval<T, Capacity> cur = pending.dequeue();

        if (gap_closed(cur.lb)) {
          converged = true;
          stop_reason = StopReason::Converged;
          stopped_early = true;
          break;
        }
        std::cout << "Least pending lb (selected for sample+partition): "
                  << cur.lb << '\n';
        GLB_ = cur.lb;

        std::vector<cu::interval<T>> box;
        cur.materialise(history, box, *policy_, var_kinds, problem.box_bounds);
        // Strictly after materialise(), which is the read this reference was
        // being held for: the parent entry is history.intervals[cur.pidx]. Once
        // the last pending child of a box has been dequeued, nothing can ever
        // look the box up again, so this is where it becomes garbage. (The
        // convergence break above skips the release, harmlessly -- the search is
        // ending.)
        history.release(cur.pidx);

        auto const assignment = policy_->choose(box, var_kinds);

        std::size_t live_count = 0;
        for (const auto& b : box) {
          if (b.ub > b.lb) {
            ++live_count;
          }
        }
        ExactGraphs* const eg =
            can_fathom_without_children(live_count, assignment.composition)
            ? find_exact_graphs(assignment.composition)
            : nullptr;

        if (eg != nullptr) {
          // Every live dimension is enumerated, so this launch's ArgMin is the
          // true best value over the remaining subtree, not just a bound --
          // fold into GUB_ and fathom, no children to enqueue.
          eg->exact.set_domain(box, assignment.var_ids);
          eg->exact.launch(/*stream=*/0);
          if (eg->exact.has_candidate()) {
            double const val = static_cast<double>(eg->exact.candidate());
            if (val < GUB_) {
              GUB_ = val;
              auto witness = eg->exact.candidate_point();
              best_point_.assign(witness.begin(), witness.end());
            }
          }
          std::cout << "iter " << iter_idx_ << ": fully enumerated and fathomed ("
                    << eg->exact.n_regions() << " points), GUB = " << GUB_
                    << '\n';
          continue;
        }

        std::size_t const box_idx = history.enqueue(box);
        CompositionGraphs& g = find_graphs(assignment.composition);

        // Best sampled feasible value from this domain; iter_idx_ salts the
        // sampler so revisited/sibling boxes draw fresh points.
        g.point.set_domain(box, assignment.var_ids, iter_idx_);
        g.point.launch(/*stream=*/0);
        double cand = static_cast<double>(g.point.candidate());
        if (cand < GUB_) {
          GUB_ = cand;
          auto witness = g.point.candidate_point();
          best_point_.assign(witness.begin(), witness.end());
        }

        // Interval analysis of sub-domains, then feasibility and GUB pruning
        g.interval.set_domain(box, assignment.var_ids);
        g.interval.launch(/*stream=*/0);
        auto obj_lb = g.interval.obj_lb();
        auto feasible = g.interval.feasible();
        for (std::size_t tid = 0; tid < g.interval.n_regions(); ++tid) {
          // feasible[tid] == 0: some constraint provably excludes its rhs over
          // this child. Sound to discard regardless of GUB_.
          if (!feasible[tid]) {
            ++pruned_infeasible;
            continue;
          }
          if (obj_lb[tid] > GUB_) {
            continue;
          }

          pending.enqueue(CompositionInterval<T, Capacity> {
              .sidx = tid,
              .pidx = box_idx,
              .depth = cur.depth + 1,
              .lb = obj_lb[tid],
              // Recorded so materialise() can prove the policy re-derives the
              // same assignment when this child is dequeued.
              .slot_count = assignment.composition.count,
          });
          // This child now names box_idx as its parent, and holds the box alive
          // until it is itself dequeued or evicted.
          history.add_ref(box_idx);
        }

        // The construction reference goes here, so a node all of whose children
        // were pruned frees its box immediately rather than retaining it for the
        // rest of the run.
        history.release(box_idx);

        std::cout << "iter " << iter_idx_ << ": GUB = " << GUB_
                  << ", Candidate: " << cand << '\n';

        // Once per iteration, after the children are in: a per-enqueue check
        // would put a branch in the hot loop to catch an overshoot that is
        // bounded by one fan-out of 40-byte nodes anyway.
        if (host_budget != 0
            && pending.capacity_bytes() + history.live_bytes() > host_budget)
        {
          if (!compacting) {
            // The default. Nothing is discarded, so the bracket below is the
            // one this search had already earned -- the run simply ends here
            // instead of growing into the OOM killer.
            std::cout
                << "Host budget of " << host_budget << " bytes reached ("
                << pending.size()
                << " regions pending); stopping the search here with the "
                   "frontier intact. The bracket below is valid. Pass "
                   "--bounded-frontier to discard the worst-bounded regions "
                   "and keep searching instead.\n";
            stop_reason = StopReason::HostMemory;
            stopped_early = true;
            break;
          }

          std::size_t const before = pending.size();
          std::size_t const n_keep = compact_keep_count(
              host_budget, history.live_bytes(), sizeof(Node));
          pending.compact(n_keep,
                          [&](const Node& e)
                          {
                            dropped.fold(static_cast<double>(e.lb), GUB_);
                            history.release(e.pidx);
                          });
          std::size_t const live =
              pending.capacity_bytes() + history.live_bytes();
          std::cout << "Host budget reached: compacted the frontier from "
                    << before << " to " << pending.size() << " regions ("
                    << live << " of " << host_budget << " bytes live)\n";

          if (live > host_budget) {
            // Live history alone crowds out the frontier, or one box is simply
            // too big to hold. Either way there is nothing left to give back, so
            // stop -- with the bracket, the frontier and the dropped floor all
            // intact -- rather than grow into the OOM killer.
            std::cout << "Host budget still exceeded with the frontier "
                         "compacted; stopping the search here. The bracket "
                         "below is valid, and the regions dropped to get under "
                         "the budget are accounted for in it.\n";
            stop_reason = StopReason::HostMemory;
            stopped_early = true;
            break;
          }
        }
      }
    } catch (const cuminlp::ResourceExhausted& e) {
      // Costed advice from GraphReplay::build (a Composition first reached
      // mid-run needs graphs that do not fit), so it is printed rather than
      // swallowed.
      std::cout << "Out of device memory mid-search: " << e.what() << '\n';
      stop_reason = StopReason::DeviceMemory;
      stopped_early = true;
    } catch (const std::bad_alloc& e) {
      std::cout << "Host allocation failed mid-search: " << e.what() << '\n';
      stop_reason = StopReason::AllocationFailure;
      stopped_early = true;
    }

    if (!stopped_early) {
      stop_reason = pending.empty() ? StopReason::Exhausted
          : gap_closed(GLB_)       ? StopReason::Converged
                                   : StopReason::IterationLimit;
    }

    bool const found_incumbent = GUB_ < std::numeric_limits<double>::max();

    // Regions not yet proven suboptimal against the final GUB_. Computed
    // before the bounds are finalised, because whether the incumbent is
    // *proven* optimal is exactly the question "is this zero".
    std::size_t const viable = pending.count_viable(GUB_);

    // Three ways to have proved it, not two, and one way to have lost the
    // right to say so. `converged` is the loop noticing that the least pending
    // lb already exceeds GUB_; an empty frontier is the same thing having
    // consumed the queue. But hitting the iteration limit with every remaining
    // region already dominated proves just as much, and used to be reported as
    // an unfinished run: nvs09 at 1248 iterations left 5825 pending regions,
    // all of them dominated, and printed a *lower* bound above its own
    // incumbent (-43.1244 <= min <= -43.1343) plus a RESULT line whose dual was
    // on the wrong side of its primal. tools/minlp_status.py records that line,
    // so the inversion did not stay cosmetic.
    //
    // The fourth term is the frontier bound: a region evicted while it could
    // still have held the optimum caps every claim below at its lb. When
    // nothing viable was ever evicted the floor is +inf and all three claims
    // read exactly as they did before this existed -- including for a
    // memory-capped run whose evictions were all dominated, which is
    // deliberately still proven optimal. See host_budget.hpp's
    // finalise_bounds.
    FinalBounds const outcome =
        finalise_bounds(GLB_,
                        GUB_,
                        tolerance_,
                        found_incumbent,
                        converged,
                        pending.empty(),
                        pending.empty() ? std::numeric_limits<double>::max()
                                        : static_cast<double>(pending.peek().lb),
                        viable,
                        dropped);
    bool const proven_optimal = outcome.proven_optimal;
    GLB_ = outcome.glb;

    if (outcome.infeasible) {
      // Frontier emptied by feasibility pruning without ever sampling a
      // feasible point -- not convergence, so GLB_ is left where the loop had
      // it rather than collapsed onto GUB_.
      //
      // This is the infeasible case, and it is worth saying so outright: it
      // cannot be reached by a sampler that merely kept missing. A region is
      // discarded only when interval analysis proves no point in it satisfies
      // some constraint, or when every one of its points was enumerated and
      // none was feasible, so a feasible instance always leaves something
      // pending. Sampling that never lands on a feasible point -- the ordinary
      // fate of an equality, whose feasible set has measure zero -- instead
      // keeps partitioning until the iteration limit stops it, and ends with a
      // nonzero pending size.
      //
      // Eviction is the third way to empty a frontier, and it proves nothing
      // at all, so finalise_bounds withholds this branch entirely once
      // anything viable has been dropped: those regions are unexplored places,
      // not excluded ones.
      //
      // "as far as interval analysis can tell" is the whole of the hedge: the
      // pruning is conservative, but it is arithmetic on rounded bounds and
      // not a certificate.
      std::cout << "Search space exhausted: every region was proven to hold no "
                   "feasible point, so the problem is infeasible as far as "
                   "interval analysis can tell. Sampling that merely never "
                   "landed on a feasible point would have left regions pending "
                   "here instead.\n";
    }

    std::cout << "------------ Finished ------------" << '\n'
              << GLB_ << " <= min <= " << GUB_ << '\n';
    if (proven_optimal) {
      std::cout << "Proven optimal: no pending region can beat the incumbent"
                << (iter_idx_ >= iter_limit_
                        ? " (the iteration limit was reached, but every"
                          " remaining region is already dominated)"
                        : "")
                << ".\n";
    }
    // An interface, not just a trace line: tools/minlp_status.py reads these to
    // tell an infeasible instance (zero pending, no drops, no incumbent) from a
    // run that stopped with places left to look, and records the two
    // differently. Printed unconditionally, so an absent line means an old
    // build rather than a zero.
    std::cout << "Stop reason: " << to_string(stop_reason) << '\n';
    std::cout << "Pending size: " << pending.size() << '\n';
    std::cout << "Viable regions: " << viable << '\n';
    std::cout << "Pruned as interval-infeasible: " << pruned_infeasible << '\n';
    std::cout << "Dropped viable regions: " << dropped.viable << '\n';
    std::cout << "Dropped dominated regions: " << dropped.dominated << '\n';
    // `none` rather than `inf`: the only meaning of an absent floor is
    // "absent", and minlp_status.py's parse_number already reads that word.
    std::cout << "Dropped lb floor: ";
    if (dropped.lost_nothing()) {
      std::cout << "none\n";
    } else {
      std::cout << dropped.lb_min << '\n';
    }
    std::cout.precision(prev_precision);
    return GUB_;
  }

private:
  std::vector<T> best_point_;
  std::shared_ptr<const CompositionPolicy<T, Capacity>> policy_;
  std::size_t sample_points_;
  std::size_t budget_bytes_;
  std::size_t host_budget_bytes_;
  FrontierPolicy frontier_policy_;
};

}  // namespace cuminlp
