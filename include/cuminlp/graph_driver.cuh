#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
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
  explicit GraphDriver(
      std::shared_ptr<const CompositionPolicy<T, Capacity>> policy,
      uint32_t iter_limit = 1000000,
      double tolerance = 1e-9,
      std::size_t sample_points = 1,
      std::size_t budget_bytes = 0)
      : driver(iter_limit, tolerance)
      , policy_(std::move(policy))
      , sample_points_(sample_points)
      , budget_bytes_(budget_bytes)
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

    std::span<const dag::VarKind> const var_kinds = problem.var_kinds;
    FanOutSpec const& fan_out = policy_->fan_out();

    // One (point, interval) replay pair per Composition actually encountered,
    // built lazily and cached -- eagerly building every possible_compositions()
    // entry could waste enormous GPU memory on graphs nothing ever launches.
    struct CompositionGraphs
    {
      Composition<Capacity> composition;
      dag::PointGraphReplay<T, Capacity> point;
      dag::IntervalGraphReplay<T, Capacity> interval;
    };

    std::vector<CompositionGraphs> graphs;
    auto find_graphs =
        [&graphs, &problem, &fan_out, this](
            const Composition<Capacity>& composition) -> CompositionGraphs&
    {
      for (auto& g : graphs) {
        if (g.composition == composition) {
          return g;
        }
      }
      graphs.push_back(CompositionGraphs {
          composition,
          dag::PointGraphReplay<T, Capacity>::build(
              problem, composition, fan_out, budget_bytes_, sample_points_),
          // sample_points_ reaches the interval graph only so its
          // out-of-memory report can cost a suggested cap against the point
          // graph too; build() clamps its own allocation to one element per
          // region regardless (GraphReplay::build, `is_point && !Exact`).
          dag::IntervalGraphReplay<T, Capacity>::build(
              problem, composition, fan_out, budget_bytes_, sample_points_),
      });
      return graphs.back();
    };

    // Same lazy/cached approach, one ExactGraphReplay per fully-enumerable
    // Composition actually encountered.
    struct ExactGraphs
    {
      Composition<Capacity> composition;
      dag::ExactGraphReplay<T, Capacity> exact;
    };

    std::vector<ExactGraphs> exact_graphs;
    auto find_exact_graphs =
        [&exact_graphs, &problem, &fan_out, this](
            const Composition<Capacity>& composition) -> ExactGraphs*
    {
      if (!is_fully_enumerable(composition)) {
        return nullptr;
      }
      for (auto& g : exact_graphs) {
        if (g.composition == composition) {
          return &g;
        }
      }
      exact_graphs.push_back(ExactGraphs {
          composition,
          // Likewise for the exact graph: reporting only, never allocation.
          dag::ExactGraphReplay<T, Capacity>::build(
              problem, composition, fan_out, budget_bytes_, sample_points_),
      });
      return &exact_graphs.back();
    };

    IntervalPQueue<T, CompositionInterval<T, Capacity>> pending(1000);
    IntervalHistory<T> history;

    std::vector<cu::interval<T>> origin {};
    history.enqueue(origin);

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

    while (iter_idx_ < iter_limit_ && !pending.empty()
           && GUB_ - GLB_ > tolerance_)
    {
      ++iter_idx_;

      CompositionInterval<T, Capacity> cur = pending.dequeue();

      if (cur.lb > GUB_) {
        converged = true;
        break;
      }
      std::cout << "Least pending lb (selected for sample+partition): "
                << cur.lb << '\n';
      GLB_ = cur.lb;

      std::vector<cu::interval<T>> box;
      cur.materialise(history, box, *policy_, var_kinds, problem.box_bounds);

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
      }

      std::cout << "iter " << iter_idx_ << ": GUB = " << GUB_
                << ", Candidate: " << cand << '\n';
    }

    bool const found_incumbent = GUB_ < std::numeric_limits<double>::max();

    if ((converged || pending.empty()) && found_incumbent) {
      // Fully explored: everything else pruned or dominated, so GUB_ is
      // optimal.
      GLB_ = GUB_;
    } else if (pending.empty()) {
      // Frontier emptied by feasibility pruning without ever sampling a
      // feasible point -- not convergence, so don't collapse GLB_ onto GUB_.
      std::cout << "Search space exhausted with no feasible point sampled; either the "
                   "problem is infeasible or point sampling never satisfied the "
                   "constraints (likely for equalities).\n";
    } else {
      GLB_ = pending.peek().lb;
    }

    // Regions not yet proven suboptimal against the final GUB_.
    std::size_t const viable = pending.count_viable(GUB_);

    std::cout << "------------ Finished ------------" << '\n'
              << GLB_ << " <= min <= " << GUB_ << '\n';
    std::cout << "Pending size: " << pending.size() << '\n';
    std::cout << "Viable regions: " << viable << '\n';
    std::cout << "Pruned as interval-infeasible: " << pruned_infeasible << '\n';

    if (found_incumbent) {
      std::streamsize const prev_precision = std::cout.precision(12);
      std::cout << "Argmin (sampled witness for GUB):" << '\n';
      for (std::size_t i = 0; i < best_point_.size(); ++i) {
        std::cout << "x[" << i << "], ";
      }
      std::cout << '\n' << "[";
      for (std::size_t i = 0; i < best_point_.size(); ++i) {
        std::cout << best_point_[i] << ", ";
      }
      std::cout << "]" << '\n';
      std::cout.precision(prev_precision);
    }
    return GUB_;
  }

private:
  std::vector<T> best_point_;
  std::shared_ptr<const CompositionPolicy<T, Capacity>> policy_;
  std::size_t sample_points_;
  std::size_t budget_bytes_;
};

}  // namespace cuminlp
