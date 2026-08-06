#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/host_budget.hpp"
#include "cuminlp/policy_catalogue.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search.hpp"
#include "cuminlp/search/cache.hpp"

namespace cuminlp
{

/// What one solve() found, and what it proved. Everything a caller used to
/// reach for through accessors on a shared base class, plus the counters the
/// run's summary quotes (design/MODULE_REFACTOR.md §6.1).
struct SearchCounters
{
  std::uint32_t iterations = 0;
  std::size_t pending = 0;
  std::size_t viable = 0;  ///< pending regions not proven suboptimal
  std::size_t pruned_infeasible = 0;
  DropAccounting dropped;
  std::size_t cache_evictions = 0;
};

template<typename T>
struct SolveOutcome
{
  double lower_bound = std::numeric_limits<double>::lowest();
  double upper_bound = std::numeric_limits<double>::max();
  std::vector<T> best_point;  ///< the witness for upper_bound; empty if none
  StopReason stop_reason = StopReason::IterationLimit;
  bool proven_optimal = false;
  bool infeasible = false;
  SearchCounters counters;

  bool found_incumbent() const
  {
    return upper_bound < std::numeric_limits<double>::max();
  }
};

// Branch-and-bound over an arbitrary dag::Problem. Each iteration, the
// CompositionPolicy picks a SlotAssignment for the current node's box, and
// the backend supplies the roles that act on it: a sampler for GUB
// candidates and a bounder for sound lower bounds / pruning. BackendCache
// owns when those get built and when they are given back.
//
// If the chosen Composition is fully enumerable and every live dimension has
// a slot, the backend's enumerator evaluates it exactly in one shot and the
// node is fathomed directly instead of enqueueing children for pruning.
//
// Names no backend, and includes no CUDA: this header compiles under a plain
// host compiler (§3.1), and the application picks the RegionBackendFactory.
template<typename T>
class SearchDriver
{
public:
  /**
   * @brief Construct a new search driver
   *
   * @param policy determines the order and partitioning/enumeration of variables
   * @param backend supplies the sampler/bounder/enumerator roles
   * @param iter_limit iteration limit (number of domains)
   * @param tolerance tolerance for primal/dual convergence check
   * @param sample_points number of points to sample per subdomain
   * @param budget_bytes caps device memory per build
   * @param host_budget_bytes caps the pending frontier and live interval history
   * @param frontier_policy policy for handling frontier growth
   * @param observer where solve()'s progress narration goes; the base
   *        class's every override is a no-op, so the default is a silent
   *        driver (design/MODULE_REFACTOR.md §8)
   */
  explicit SearchDriver(
      std::shared_ptr<const CompositionPolicy<T>> policy,
      std::shared_ptr<const backend::RegionBackendFactory<T>> backend,
      uint32_t iter_limit = 1000000,
      double tolerance = 1e-6,
      std::size_t sample_points = 1,
      std::size_t budget_bytes = 0,
      std::size_t host_budget_bytes = 0,
      FrontierPolicy frontier_policy = FrontierPolicy::StopAtBudget,
      std::shared_ptr<report::SearchObserver> observer =
          std::make_shared<report::SearchObserver>())
      : policy_(std::move(policy))
      , backend_(std::move(backend))
      , tolerance_(tolerance)
      , iter_limit_(iter_limit)
      , sample_points_(sample_points)
      , budget_bytes_(budget_bytes)
      , host_budget_bytes_(host_budget_bytes)
      , frontier_policy_(frontier_policy)
      , observer_(std::move(observer))
  {
    if (policy_ == nullptr) {
      throw cuminlp::InvalidConfiguration(
          "SearchDriver requires a non-null CompositionPolicy");
    }
    if (backend_ == nullptr) {
      throw cuminlp::InvalidConfiguration(
          "SearchDriver requires a non-null RegionBackendFactory");
    }
    if (observer_ == nullptr) {
      throw cuminlp::InvalidConfiguration(
          "SearchDriver requires a non-null SearchObserver");
    }
  }

  // the driver loop
  auto solve(const dag::Problem<T>& problem) -> SolveOutcome<T>
  {
    using search::IntervalHistory;
    using search::IntervalPQueue;
    using search::Node;

    GUB_ = std::numeric_limits<double>::max();
    GLB_ = std::numeric_limits<double>::lowest();
    iter_idx_ = 0;
    best_point_.clear();

    std::streamsize const prev_precision = std::cout.precision(17);

    std::span<const dag::VarKind> const var_kinds = problem.var_kinds;
    FanOutSpec const& fan_out = policy_->fan_out();

    // Roles per Composition, built on first use and evicted under pressure
    // -- see search/cache.hpp for why that is the search layer's decision
    // and not the backend's.
    search::BackendCache<T> cache(
        backend_, problem, fan_out, budget_bytes_, sample_points_, *observer_);

    IntervalPQueue<T> pending(10000);
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
    observer_->on_start(host_budget, host_budget_bytes_ != 0, compacting);

    pending.enqueue(Node<T> {
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
    DropAccounting dropped;

    // Set at every break out of the loop below; when nothing breaks, the loop
    // condition says why it ended and the reason is derived after it.
    StopReason stop_reason = StopReason::IterationLimit;
    bool stopped_early = false;

    auto const gap_closed = [this](double bound) {
      return cuminlp::gap_closed(GUB_, bound, tolerance_);
    };

    // The loop is wrapped so a failed allocation ends the run through the same
    // epilogue as a spent iteration budget: the bracket GLB_ <= min <= GUB_ is
    // valid at every iteration, and letting the exception out of solve() would
    // throw it away. A SIGKILL still fails abnormally.
    try {
      while (iter_idx_ < iter_limit_ && !pending.empty() && !gap_closed(GLB_)) {
        ++iter_idx_;

        Node<T> cur = pending.dequeue();

        if (gap_closed(cur.lb)) {
          converged = true;
          stop_reason = StopReason::Converged;
          stopped_early = true;
          break;
        }
        observer_->on_dequeue(iter_idx_, static_cast<double>(cur.lb));
        GLB_ = cur.lb;

        std::vector<cu::interval<T>> box;
        cur.materialise(history, box, *policy_, var_kinds, problem.box_bounds);
        history.release(cur.pidx);

        auto const assignment = policy_->choose(box, var_kinds);
        assert(assignment_is_distinct_and_live<T>(assignment, box)
               && "CompositionPolicy::choose returned duplicate or "
                  "non-live var_ids (design/MODULE_REFACTOR.md §4.5)");

        std::size_t live_count = 0;
        for (const auto& b : box) {
          if (b.ub > b.lb) {
            ++live_count;
          }
        }
        backend::RegionEnumerator<T>* const enumerator =
            can_fathom_without_children(live_count, assignment.composition)
            ? cache.enumerator(assignment.composition)
            : nullptr;

        backend::Region<T> const region {box, assignment};

        if (enumerator != nullptr) {
          // Every live dimension is enumerated, so this launch's ArgMin is the
          // true best value over the remaining subtree, not just a bound --
          // fold into GUB_ and fathom, no children to enqueue.
          auto const exact = enumerator->enumerate(region);
          if (exact.found) {
            double const val = static_cast<double>(exact.value);
            if (val < GUB_) {
              GUB_ = val;
              best_point_.assign(exact.point.begin(), exact.point.end());
              observer_->on_incumbent(GUB_);
            }
          }
          observer_->on_fathom(enumerator->n_points(), GUB_);
          continue;
        }

        std::size_t const box_idx = history.enqueue(box);
        backend::SubdivisionBundle<T>& roles =
            cache.subdivision(assignment.composition);

        // Best sampled feasible value from this domain; iter_idx_ salts the
        // sampler so revisited/sibling boxes draw fresh points.
        auto const drawn = roles.sampler->sample(region, iter_idx_);
        double cand = static_cast<double>(drawn.value);
        if (cand < GUB_) {
          GUB_ = cand;
          best_point_.assign(drawn.point.begin(), drawn.point.end());
          observer_->on_incumbent(GUB_);
        }

        // Interval analysis of sub-domains, then feasibility and GUB pruning
        auto const bounds = roles.bounder->bound(region);
        auto obj_lb = bounds.obj_lb;
        auto feasible = bounds.feasible;
        for (std::size_t tid = 0; tid < bounds.n_regions; ++tid) {
          // feasible[tid] == 0: some constraint provably excludes its rhs over
          // this child. Sound to discard regardless of GUB_.
          if (!feasible[tid]) {
            ++pruned_infeasible;
            continue;
          }
          if (obj_lb[tid] > GUB_) {
            continue;
          }

          pending.enqueue(Node<T> {
              .sidx = tid,
              .pidx = box_idx,
              .depth = cur.depth + 1,
              .lb = obj_lb[tid],
              // Recorded so materialise() can prove the policy re-derives the
              // same assignment when this child is dequeued.
              .slot_count = assignment.composition.size(),
          });
          // This child now names box_idx as its parent, and holds the box alive
          // until it is itself dequeued or evicted.
          history.add_ref(box_idx);
        }

        history.release(box_idx);

        observer_->on_iteration(report::IterationEvent {.gub = GUB_,
                                                        .candidate = cand});

        // Once per iteration, after the children are in: a per-enqueue check
        // would put a branch in the hot loop to catch an overshoot that is
        // bounded by one fan-out of 40-byte nodes anyway.
        if (over_host_budget(pending.capacity_bytes() + history.live_bytes(),
                             host_budget))
        {
          if (!compacting) {
            // The default. Nothing is discarded, so the bracket below is the
            // one this search had already earned -- the run simply ends here
            // instead of growing into the OOM killer.
            observer_->on_budget_stop(
                report::BudgetStopEvent {.after_compaction = false,
                                         .host_budget = host_budget,
                                         .pending_size = pending.size()});
            stop_reason = StopReason::HostMemory;
            stopped_early = true;
            break;
          }

          std::size_t const before = pending.size();
          std::size_t const n_keep = compact_keep_count(
              host_budget, history.live_bytes(), sizeof(Node<T>));
          pending.compact(n_keep,
                          [&](const Node<T>& e)
                          {
                            dropped.fold(static_cast<double>(e.lb), GUB_);
                            history.release(e.pidx);
                          });
          std::size_t const live =
              pending.capacity_bytes() + history.live_bytes();
          observer_->on_compaction(
              report::CompactionEvent {.before = before,
                                       .after = pending.size(),
                                       .live_bytes = live,
                                       .host_budget = host_budget});

          if (over_host_budget(live, host_budget)) {
            // Live history alone crowds out the frontier, or one box is simply
            // too big to hold. Either way there is nothing left to give back, so
            // stop -- with the bracket, the frontier and the dropped floor all
            // intact -- rather than grow into the OOM killer.
            observer_->on_budget_stop(
                report::BudgetStopEvent {.after_compaction = true,
                                         .host_budget = host_budget,
                                         .pending_size = pending.size()});
            stop_reason = StopReason::HostMemory;
            stopped_early = true;
            break;
          }
        }
      }
    } catch (const backend::OverBudgetError& e) {
      // A Composition first reached mid-run needs graphs that do not fit. The
      // backend threw the facts; the costed advice is the resolver's to write
      // (design/MODULE_REFACTOR.md §5.6), and is reported rather than
      // swallowed -- ConsoleReporter's on_backend_error costs it against the
      // ProblemProfile it was constructed with.
      observer_->on_backend_error(e.facts());
      stop_reason = StopReason::DeviceMemory;
      stopped_early = true;
    } catch (const cuminlp::ResourceExhausted& e) {
      observer_->on_mid_search_error(
          std::string("Out of device memory mid-search: ") + e.what());
      stop_reason = StopReason::DeviceMemory;
      stopped_early = true;
    } catch (const std::bad_alloc& e) {
      observer_->on_mid_search_error(
          std::string("Host allocation failed mid-search: ") + e.what());
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

    // outcome.infeasible: frontier emptied by feasibility pruning without
    // ever sampling a feasible point -- not convergence, so GLB_ is left
    // where the loop had it rather than collapsed onto GUB_. Eviction is the
    // third way to empty a frontier and proves nothing, so finalise_bounds
    // withholds this branch entirely once anything viable has been dropped.
    //
    // stop_reason/pending/viable/pruned/dropped: not just a trace, an
    // interface -- tools/minlp_status.py reads these to tell an infeasible
    // instance (zero pending, no drops, no incumbent) from a run that
    // stopped with places left to look, and records the two differently.
    // ConsoleReporter prints them unconditionally, so an absent line means
    // an old build rather than a zero.
    observer_->on_finish(report::FinalReport {
        .glb = GLB_,
        .gub = GUB_,
        .proven_optimal = proven_optimal,
        .infeasible = outcome.infeasible,
        .stop_reason = stop_reason,
        .pending_size = pending.size(),
        .viable = viable,
        .pruned_infeasible = pruned_infeasible,
        .dropped = dropped,
        .iter_idx = iter_idx_,
        .iter_limit = iter_limit_,
    });
    std::cout.precision(prev_precision);

    return SolveOutcome<T> {
        .lower_bound = GLB_,
        .upper_bound = GUB_,
        .best_point = best_point_,
        .stop_reason = stop_reason,
        .proven_optimal = proven_optimal,
        .infeasible = outcome.infeasible,
        .counters =
            SearchCounters {
                .iterations = iter_idx_,
                .pending = pending.size(),
                .viable = viable,
                .pruned_infeasible = pruned_infeasible,
                .dropped = dropped,
                .cache_evictions = cache.evictions(),
            },
    };
  }

private:
  std::shared_ptr<const CompositionPolicy<T>> policy_;
  std::shared_ptr<const backend::RegionBackendFactory<T>> backend_;
  double tolerance_;
  std::uint32_t iter_limit_;
  std::size_t sample_points_;
  std::size_t budget_bytes_;
  std::size_t host_budget_bytes_;
  FrontierPolicy frontier_policy_;
  std::shared_ptr<report::SearchObserver> observer_;

  // The bracket the loop maintains: GLB_ <= min <= GUB_ at every iteration.
  // Members rather than locals only so gap_closed() can close over them;
  // solve() resets all four on entry, so a driver is reusable.
  double GUB_ = std::numeric_limits<double>::max();
  double GLB_ = std::numeric_limits<double>::lowest();
  std::uint32_t iter_idx_ = 0;
  std::vector<T> best_point_;
};

}  // namespace cuminlp
