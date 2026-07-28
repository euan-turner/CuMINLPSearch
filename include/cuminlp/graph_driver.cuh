#pragma once

#include <cstddef>
#include <iostream>
#include <span>
#include <vector>

#include "cuminlp/cuda_utils.cuh"
#include "cuminlp/cuminlp.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/graph_replay.cuh"
#include "cuminlp/search.hpp"

namespace cuminlp
{

// Driver for an arbitrary dag::Problem, evaluated each
// iteration by two CUDA-graph replays (graph_replay.cuh) built once from the
// problem's expression DAG: a point-sample graph for GUB candidates and an
// interval graph for sound lower bounds / pruning. CycleSize dimensions are
// partitioned into PartitionNum slices per iteration (PartitionNum^CycleSize
// children), each sampled at SamplePoints points for its GUB candidate.
// Compare FixedRosenbrockDriver (fixed_rosenbrock_driver.hpp), which hardcodes
// its problem and evaluates via hand-written kernels instead.
template<typename T, std::size_t CycleSize, std::size_t PartitionNum, std::size_t SamplePoints>
class GraphDriver : public driver
{
public:
  using driver::driver;

  auto solve(const dag::Problem<T>& problem) -> double
  {
    using search::CompressedInterval;
    using search::IntervalHistory;
    using search::IntervalPQueue;

    std::size_t const dims = problem.box_bounds.size();
    std::size_t const num_children = detail::ipow<PartitionNum, CycleSize>();

    // Built once, replayed every iteration against a new box/cycle_start
    auto point_replay = dag::PointGraphReplay<T, CycleSize, PartitionNum, SamplePoints>::build(
        problem, num_children);
    auto interval_replay =
        dag::IntervalGraphReplay<T, CycleSize, PartitionNum>::build(problem, num_children);

    IntervalPQueue<T> pending(1000);
    IntervalHistory<T> history;

    std::vector<cu::interval<T>> origin {};
    history.enqueue(origin);

    pending.enqueue(CompressedInterval<T> {
        .sidx = 0,
        .pidx = 0,
        .depth = 0,
        .cycle_start = 0,
        .lb = GLB_,
    });

    bool converged = false;

    while (iter_idx_ < iter_limit_ && !pending.empty() && GUB_ - GLB_ > tolerance_)
    {
      ++iter_idx_;

      CompressedInterval<T> cur = pending.dequeue();

      if (cur.lb > GUB_) {
        converged = true;
        break;
      }
      std::cout << "Least pending lb (selected for sample+partition): " << cur.lb << '\n';
      GLB_ = cur.lb;

      std::vector<cu::interval<T>> box;
      box.resize(dims);
      if (cur.pidx == 0) {
        box = problem.box_bounds;
      } else {
        cur.materialise(history, box, CycleSize, PartitionNum);
      }

      std::size_t const box_idx = history.enqueue(box);
      std::size_t const child_cycle_start = (cur.cycle_start + CycleSize) % dims;

      // GUB sampling: candidate is the best sampled, feasible value from this domain
      point_replay.set_domain(box, child_cycle_start);
      point_replay.launch(/*stream=*/0);
      double cand = static_cast<double>(point_replay.candidate());
      GUB_ = std::min(GUB_, cand);

      // Interval analysis of sub-domains and GUB pruning
      interval_replay.set_domain(box, child_cycle_start);
      interval_replay.launch(/*stream=*/0);
      auto obj_lb = interval_replay.obj_lb();
      for (std::size_t tid = 0; tid < num_children; ++tid) {
        if (obj_lb[tid] > GUB_) continue;

        pending.enqueue(CompressedInterval<T> {
            .sidx = tid,
            .pidx = box_idx,
            .depth = cur.depth + 1,
            .cycle_start = child_cycle_start,
            .lb = obj_lb[tid],
        });
      }

      std::cout << "iter " << iter_idx_ << ": GUB = " << GUB_  << ", Candidate: " << cand << '\n';
    }

    if (converged || pending.empty()) {
      // Fully explored: every remaining possibility has been pruned or
      // proven dominated, so the best upper bound found is proven optimal.
      GLB_ = GUB_;
    } else {
      // GLB is smallest lower bound on any pending region, which is the first
      GLB_ = pending.peek().lb;
    }

    // Regions not yet proven suboptimal against the final GUB_, i.e. regions
    // that could still contain the global optimum.
    std::size_t const viable = pending.count_viable(GUB_);

    std::cout << "------------ Finished ------------" << '\n'
              << GLB_ << " <= min <= " << GUB_ << '\n';
    std::cout << "Pending size: " << pending.size() << '\n';
    std::cout << "Viable regions: " << viable << '\n';
    return GUB_;
  }
};

}  // namespace cuminlp
