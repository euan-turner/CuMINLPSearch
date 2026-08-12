#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/aggregate/bound.hpp"
#include "cuminlp/aggregate/frontier.hpp"
#include "cuminlp/aggregate/partition.hpp"
#include "cuminlp/aggregate/policy.hpp"
#include "cuminlp/aggregate/selection.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/eval.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/report/telemetry.hpp"
#include "cuminlp/search/budget.hpp"
#include "cuminlp/search/epilogue.hpp"
#include "cuminlp/search/history.hpp"

// The aggregate search loop (design/AGGREGATE_BOUNDING.md §3.1).
//
// A separate driver from search::SearchDriver rather than a parameterisation
// of it: the per-child prune/enqueue loop over `N` regions collapses to `k`
// unconditional ones, the fathom branch changes meaning, and the frontier
// answers two questions instead of one. Two readable loops beat one shared
// skeleton carrying both shapes.
//
// What *is* shared, unchanged: search::IntervalHistory, search::budget.hpp,
// search::finalise_bounds, and the whole report/ observer fan -- including
// ConsoleReporter, so that the `Outcome:` and summary lines
// tools/minlp_status.py scrapes are byte-identical between the two backends.
//
// Names no backend and includes no CUDA: like search/driver.hpp, this header
// compiles under a plain host compiler, and the application supplies the
// roles.
namespace cuminlp::aggregate
{

template<typename T>
struct AggregateOutcome
{
  double lower_bound = std::numeric_limits<double>::lowest();
  double upper_bound = std::numeric_limits<double>::max();
  std::vector<T> best_point;
  search::StopReason stop_reason = search::StopReason::IterationLimit;
  bool proven_optimal = false;
  bool infeasible = false;
  report::RunTelemetry counters;

  /// Whether the run ever left `RejectIndex`'s pre-incumbent fallback (§7.5).
  /// Reported rather than assumed: on an equality-heavy model the sampler may
  /// never produce an incumbent, in which case a run recorded as
  /// `reject-index` in fact spent all of itself ordering by `lb`.
  bool selection_rule_engaged = false;

  bool found_incumbent() const
  {
    return upper_bound < std::numeric_limits<double>::max();
  }
};

/// Supplies the two device roles for a launch composition.
///
/// The implementation holds one graph per role and rebuilds on a composition
/// change -- each graph is sized to fill its whole share of the device budget,
/// so two of them do not fit. See backend/aggregate/cache.cuh.
template<typename T>
class AggregateRoleCache
{
public:
  virtual ~AggregateRoleCache() = default;
  virtual AggregateBounder<T>& bounder(const region::Composition&) = 0;
  virtual AggregateSampler<T>& sampler(const region::Composition&) = 0;
  virtual std::size_t graphs_built() const = 0;

  /// Builds that replaced a live graph rather than filling an empty slot.
  /// A high count means the launch composition is churning; see
  /// GraphRoleCache's comment.
  virtual std::size_t graphs_rebuilt() const = 0;
};

/**
 * @brief Branch-and-bound in which the device bounds `k` children rather than
 *        enumerating `N`.
 */
template<typename T>
class AggregateDriver
{
public:
  /**
   * @param policy partitions a node for the bounder (`N` subregions).
   * @param sampler_policy partitions the same node for the sampler, at its
   *        own smaller budget (§6.3). Its branch half is the same shape, so
   *        each child still receives its share of strata.
   * @param known_primal_bound seeds `GUB_`; must already be sound in the
   *        minimising sense.
   */
  AggregateDriver(std::shared_ptr<const AggregatePolicy<T>> policy,
                  std::shared_ptr<const AggregatePolicy<T>> sampler_policy,
                  std::shared_ptr<AggregateRoleCache<T>> cache,
                  SelectionRule rule,
                  std::uint32_t iter_limit,
                  double tolerance,
                  std::size_t host_budget_bytes,
                  std::shared_ptr<report::SearchObserver> observer,
                  std::optional<double> known_primal_bound = std::nullopt)
      : policy_(std::move(policy))
      , sampler_policy_(std::move(sampler_policy))
      , cache_(std::move(cache))
      , rule_(rule)
      , iter_limit_(iter_limit)
      , tolerance_(tolerance)
      , host_budget_bytes_(host_budget_bytes)
      , observer_(std::move(observer))
      , known_primal_bound_(known_primal_bound)
  {
    if (policy_ == nullptr || sampler_policy_ == nullptr || cache_ == nullptr
        || observer_ == nullptr)
    {
      throw cuminlp::
          InvalidConfiguration(
              "AggregateDriver requires a non-null policy, sampler policy, "
              "cache " "and observer");
    }
  }

  AggregateOutcome<T> solve(const model::Problem<T>& problem)
  {
    double gub =
        known_primal_bound_.value_or(std::numeric_limits<double>::max());
    std::vector<T> best_point;
    std::span<const model::VarKind> const var_kinds = problem.var_kinds;
    std::size_t const k = policy_->branch_fan_out();

    AggregateFrontier<T> frontier(rule_);
    search::IntervalHistory<T> history;
    // Slot 0 is the root sentinel: materialise() answers pidx == 0 from
    // root_box without looking it up, so this entry exists only to make index
    // 0 mean "the root".
    history.enqueue(std::vector<cu::interval<T>> {});

    std::size_t const host_budget =
        search::resolve_host_budget(host_budget_bytes_);
    observer_->on_start(host_budget, host_budget_bytes_ != 0, false);

    frontier.enqueue(AggregateNode<T> {.sidx = 0,
                                       .pidx = 0,
                                       .depth = 0,
                                       .lb = std::numeric_limits<T>::lowest(),
                                       .hull_ub = std::numeric_limits<T>::max(),
                                       .branch_hash = 0},
                     gub);

    std::uint32_t iter = 0;
    bool converged = false;
    bool stopped_early = false;
    bool rule_engaged = false;
    auto stop_reason = search::StopReason::IterationLimit;
    search::DropAccounting dropped;  // nothing is ever evicted; see §10

    std::uint64_t bounded = 0;
    std::uint64_t sampled = 0;
    std::uint64_t enqueued = 0;
    std::uint64_t fathomed = 0;
    std::uint64_t pruned_infeasible = 0;
    std::uint64_t pruned_dominated = 0;
    std::uint64_t witness_rejected = 0;

    double glb = std::numeric_limits<double>::lowest();

    auto const closed = [&](double bound)
    { return search::gap_closed(gub, bound, tolerance_); };

    // Wrapped so a failed allocation or an over-budget build ends the run
    // through the same epilogue as a spent iteration budget: the bracket is
    // valid at every iteration, and letting the exception out would throw it
    // away.
    try {
      while (iter < iter_limit_ && !frontier.empty()) {
        double const frontier_lb = static_cast<double>(frontier.min_lb());
        glb = frontier_lb;
        if (closed(frontier_lb)) {
          converged = true;
          stop_reason = search::StopReason::Converged;
          stopped_early = true;
          break;
        }
        ++iter;

        AggregateNode<T> const cur = frontier.pop();
        observer_->on_dequeue(iter, static_cast<double>(cur.lb));

        std::vector<cu::interval<T>> box;
        cur.materialise(history, box, *policy_, var_kinds, problem.box_bounds);
        history.release(cur.pidx);

        AggregatePartition const part =
            policy_->partition(box, var_kinds, cur.depth);

        if (part.is_leaf()) {
          // Every variable is resolved, so this box is a single point and the
          // objective at it is exact -- no children, and no launch. The
          // existing backend reaches the same conclusion through an exact
          // enumeration graph; here the refine has already driven the box to
          // a point, so the only thing left is to evaluate it.
          ++fathomed;
          fold_point(problem, box, gub, best_point, witness_rejected, frontier);
          continue;
        }
        part.validate();

        AggregatePartition const sampler_part =
            sampler_policy_->partition(box, var_kinds, cur.depth);

        std::size_t const box_idx = history.enqueue(box);

        // Sampling first: an incumbent found here can immediately dominate a
        // child of this same node, which is a cheaper prune than discovering
        // it an iteration later.
        if (!sampler_part.is_leaf()) {
          AggregateSampler<T>& sampler =
              cache_->sampler(sampler_part.launch.composition);
          AggregateRegion<T> const sample_region {box, sampler_part};
          auto const drawn = sampler.sample({&sample_region, 1}, iter);
          sampled += sampler.n_samples();
          if (drawn.found && static_cast<double>(drawn.value) < gub) {
            if (search::witness_is_admissible<T>(
                    drawn.point, var_kinds, problem.box_bounds))
            {
              gub = static_cast<double>(drawn.value);
              best_point.assign(drawn.point.begin(), drawn.point.end());
              observer_->on_incumbent(gub);
              frontier.on_incumbent(gub);
              rule_engaged = true;
            } else {
              ++witness_rejected;
            }
          }
        }

        AggregateBounder<T>& bounder = cache_->bounder(part.launch.composition);
        AggregateRegion<T> const bound_region {box, part};
        auto const bounds = bounder.bound_children({&bound_region, 1});
        bounded += bounds.size();

        std::uint64_t const branch_hash = region::assignment_hash(part.branch);

        for (std::size_t c = 0; c < bounds.size(); ++c) {
          if (!bounds[c].feasible) {
            ++pruned_infeasible;
            continue;
          }
          // §4.4's clamp, and the reason this is not simply `bounds[c].lb`:
          // the child's cover and its parent's are different covers, not
          // nested ones, so an aggregate over a re-partitioned child can be
          // *weaker* than the parent's own bound. Sound either way -- the
          // parent's lb is a valid lower bound over the child, since the
          // child is inside it -- but without this the frontier's minimum
          // could visibly fall during a run.
          T const lb = std::max(bounds[c].lb, cur.lb);

          // The clamp can lift `lb` above this child's own `hull_ub`, and
          // when it does the child is **provably empty**, not merely
          // dominated. Both bounds are sound: `lb` holds over the child
          // because the child is inside its parent, and `hull_ub` is an upper
          // bound on the objective anywhere in the child's unexcluded part --
          // a feasible point of the child would have to lie in some
          // unexcluded subregion, since exclusion only ever fires on a
          // provably infeasible one. A feasible x would give
          // `lb <= f(x) <= hull_ub`, so `lb > hull_ub` says there is none.
          //
          // So the clamp is not only what makes the frontier's minimum
          // monotone (§4.4) -- it also pays for itself as an extra
          // feasibility test that costs nothing and needs no launch.
          if (lb > bounds[c].hull_ub) {
            ++pruned_infeasible;
            continue;
          }
          if (static_cast<double>(lb) > gub) {
            ++pruned_dominated;
            continue;
          }
          frontier.enqueue(AggregateNode<T> {.sidx = c,
                                             .pidx = box_idx,
                                             .depth = cur.depth + 1,
                                             .lb = lb,
                                             .hull_ub = bounds[c].hull_ub,
                                             .branch_hash = branch_hash},
                           gub);
          history.add_ref(box_idx);
          ++enqueued;
        }

        history.release(box_idx);
        observer_->on_iteration(
            report::IterationEvent {.gub = gub, .candidate = gub});

        if (search::over_host_budget(
                frontier.capacity_bytes() + history.live_bytes(), host_budget))
        {
          // Stop, never compact (§10). The frontier grows by k - 1 per
          // iteration, so reaching a budget at all is a finding rather than
          // an expected mode, and discarding regions to keep going would
          // change which tree was explored for no reason anyone asked for.
          observer_->on_budget_stop(
              report::BudgetStopEvent {.after_compaction = false,
                                       .host_budget = host_budget,
                                       .pending_size = frontier.size()});
          stop_reason = search::StopReason::HostMemory;
          stopped_early = true;
          break;
        }
      }
    } catch (const backend::OverBudgetError& e) {
      observer_->on_backend_error(e.facts());
      stop_reason = search::StopReason::DeviceMemory;
      stopped_early = true;
    } catch (const cuminlp::ResourceExhausted& e) {
      observer_->on_mid_search_error(
          std::string("Out of device memory mid-search: ") + e.what());
      stop_reason = search::StopReason::DeviceMemory;
      stopped_early = true;
    } catch (const std::bad_alloc& e) {
      observer_->on_mid_search_error(
          std::string("Host allocation failed mid-search: ") + e.what());
      stop_reason = search::StopReason::AllocationFailure;
      stopped_early = true;
    }

    if (!stopped_early) {
      stop_reason = frontier.empty() ? search::StopReason::Exhausted
                                     : search::StopReason::IterationLimit;
    }

    bool const found_incumbent = gub < std::numeric_limits<double>::max();
    std::size_t const viable = frontier.count_viable(static_cast<T>(gub));
    double const frontier_min = frontier.empty()
        ? std::numeric_limits<double>::max()
        : static_cast<double>(frontier.min_lb());

    // The same epilogue the existing driver uses, unchanged: which claims a
    // run has earned is a property of the bracket and what was discarded, not
    // of how the bounds were computed.
    search::FinalBounds const outcome =
        search::finalise_bounds(glb,
                                gub,
                                tolerance_,
                                found_incumbent,
                                converged,
                                frontier.empty(),
                                frontier_min,
                                viable,
                                dropped);
    glb = outcome.glb;

    report::RunTelemetry const telemetry {
        .iterations = iter,
        .fathomed = fathomed,
        .fathomed_points = fathomed,  // a leaf is exactly one point
        // Children bounded, not subregions evaluated -- that keeps
        // RunTelemetry::balances()'s first identity meaningful, and the device
        // figure is `iterations * N`, recoverable from the GRAPH line.
        .bounded = bounded,
        .sampled = sampled,
        .enqueued = enqueued,
        .witness_rejected = witness_rejected,
        .pruned_duplicate = 0,  // §4.6: the refine never produces one
        .pruned_infeasible = pruned_infeasible,
        .pruned_dominated = pruned_dominated,
        .dropped = dropped,
        .pending = frontier.size(),
        .viable = viable,
        .history_freed = history.freed_count(),
        .history_peak_live = history.peak_live(),
        .graphs_built = cache_->graphs_built(),
        .graphs_rebuilt = cache_->graphs_rebuilt(),
        .cache_evictions = 0,
    };
    observer_->on_telemetry(telemetry);

    observer_->on_finish(report::FinalReport {
        .glb = glb,
        .gub = gub,
        .proven_optimal = outcome.proven_optimal,
        .infeasible = outcome.infeasible,
        .stop_reason = stop_reason,
        .pending_size = frontier.size(),
        .viable = viable,
        .pruned_infeasible = pruned_infeasible,
        .dropped = dropped,
        .iter_idx = iter,
        .iter_limit = iter_limit_,
    });

    return AggregateOutcome<T> {
        .lower_bound = glb,
        .upper_bound = gub,
        .best_point = best_point,
        .stop_reason = stop_reason,
        .proven_optimal = outcome.proven_optimal,
        .infeasible = outcome.infeasible,
        .counters = telemetry,
        .selection_rule_engaged = rule_engaged || !rebuilds_on_incumbent(rule_),
    };
  }

private:
  /// A fully-resolved box: one point, evaluated on the host. Its objective is
  /// exact, so a feasible one is an incumbent outright rather than a bound.
  void fold_point(const model::Problem<T>& problem,
                  const std::vector<cu::interval<T>>& box,
                  double& gub,
                  std::vector<T>& best_point,
                  std::uint64_t& witness_rejected,
                  AggregateFrontier<T>& frontier)
  {
    std::vector<T> point(box.size());
    for (std::size_t i = 0; i < box.size(); ++i) {
      point[i] = box[i].lb;
    }
    if (!model::satisfies_constraints(problem, point)) {
      return;
    }
    T const value = model::evaluate_objective(problem, point);
    if (!(static_cast<double>(value) < gub)) {
      return;
    }
    if (!search::witness_is_admissible<T>(
            point, problem.var_kinds, problem.box_bounds))
    {
      ++witness_rejected;
      return;
    }
    gub = static_cast<double>(value);
    best_point = point;
    observer_->on_incumbent(gub);
    frontier.on_incumbent(gub);
  }

  std::shared_ptr<const AggregatePolicy<T>> policy_;
  std::shared_ptr<const AggregatePolicy<T>> sampler_policy_;
  std::shared_ptr<AggregateRoleCache<T>> cache_;
  SelectionRule rule_;
  std::uint32_t iter_limit_;
  double tolerance_;
  std::size_t host_budget_bytes_;
  std::shared_ptr<report::SearchObserver> observer_;
  std::optional<double> known_primal_bound_;
};

}  // namespace cuminlp::aggregate
