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
#include "cuminlp/graph_replay.cuh"
#include "cuminlp/search.hpp"

namespace cuminlp
{

// Driver for an arbitrary dag::Problem. Each iteration, the CompositionPolicy
// picks a SlotAssignment for the current node's box; the corresponding
// (point, interval) GraphReplay pair -- point for GUB candidates, interval
// for sound lower bounds / pruning -- is built lazily on first use and
// cached (find_graphs/find_exact_graphs), since possible_compositions() can
// be large and most entries may go unused for a given problem/CycleSize.
//
// If a node's live dimensions all fit within CycleSize and the chosen
// Composition is fully enumerable, an ExactGraphReplay evaluates it exactly
// in one shot and fathoms it directly instead of enqueueing children for
// interval pruning.
//
// CompositionPolicy is a runtime constructor argument (not a template
// parameter) so it can be chosen via CLI flag without recompiling.
// EnumerateCap defaults to PartitionNum; override only to decouple the
// enumerate threshold from the bisection width (must match the policy's own).
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints,
         std::size_t EnumerateCap = PartitionNum>
class GraphDriver : public driver
{
public:
  explicit GraphDriver(
      std::shared_ptr<const CompositionPolicy<T, CycleSize>> policy,
      uint32_t iter_limit = 1000000,
      double tolerance = 1e-9)
      : driver(iter_limit, tolerance)
      , policy_(std::move(policy))
  {
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

    // One (point, interval) replay pair per Composition actually encountered,
    // built lazily and cached -- eagerly building every possible_compositions()
    // entry could waste enormous GPU memory on graphs nothing ever launches.
    struct CompositionGraphs
    {
      Composition<CycleSize> composition;
      dag::PointGraphReplay<T,
                            CycleSize,
                            PartitionNum,
                            SamplePoints,
                            EnumerateCap>
          point;
      dag::IntervalGraphReplay<T, CycleSize, PartitionNum, EnumerateCap>
          interval;
    };

    std::vector<CompositionGraphs> graphs;
    auto find_graphs =
        [&graphs, &problem](
            const Composition<CycleSize>& composition) -> CompositionGraphs&
    {
      for (auto& g : graphs) {
        if (g.composition == composition) {
          return g;
        }
      }
      graphs.push_back(CompositionGraphs {
          composition,
          dag::PointGraphReplay<T,
                                CycleSize,
                                PartitionNum,
                                SamplePoints,
                                EnumerateCap>::build(problem, composition),
          dag::IntervalGraphReplay<T, CycleSize, PartitionNum, EnumerateCap>::
              build(problem, composition),
      });
      return graphs.back();
    };

    // Same lazy/cached approach, one ExactGraphReplay per fully-enumerable
    // Composition actually encountered.
    struct ExactGraphs
    {
      Composition<CycleSize> composition;
      dag::ExactGraphReplay<T, CycleSize, PartitionNum, EnumerateCap> exact;
    };

    std::vector<ExactGraphs> exact_graphs;
    auto find_exact_graphs =
        [&exact_graphs,
         &problem](const Composition<CycleSize>& composition) -> ExactGraphs*
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
          dag::ExactGraphReplay<T, CycleSize, PartitionNum, EnumerateCap>::
              build(problem, composition),
      });
      return &exact_graphs.back();
    };

    IntervalPQueue<
        T,
        CompositionInterval<T, CycleSize, PartitionNum, EnumerateCap>>
        pending(1000);
    IntervalHistory<T> history;

    std::vector<cu::interval<T>> origin {};
    history.enqueue(origin);

    pending.enqueue(
        CompositionInterval<T, CycleSize, PartitionNum, EnumerateCap> {
            .sidx = 0,
            .pidx = 0,
            .depth = 0,
            .lb = GLB_,
        });

    bool converged = false;
    std::size_t pruned_infeasible = 0;

    while (iter_idx_ < iter_limit_ && !pending.empty()
           && GUB_ - GLB_ > tolerance_)
    {
      ++iter_idx_;

      CompositionInterval<T, CycleSize, PartitionNum, EnumerateCap> cur =
          pending.dequeue();

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

        pending.enqueue(
            CompositionInterval<T, CycleSize, PartitionNum, EnumerateCap> {
                .sidx = tid,
                .pidx = box_idx,
                .depth = cur.depth + 1,
                .lb = obj_lb[tid],
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
  std::shared_ptr<const CompositionPolicy<T, CycleSize>> policy_;
};

}  // namespace cuminlp
