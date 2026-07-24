#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include "cuminlp/cuminlp.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/graph_replay.cuh"
#include "cuminlp/rosenbrock.cuh"
#include "cuminlp/search.hpp"

using namespace cuminlp::dag;

namespace cuminlp::rosenbrock
{

Problem<double> create_rosenbrock(int num_vars) {
  Problem<double> p;
  std::vector<Expr<double>> x;
  x.reserve(num_vars);
  for (int i = 0; i < num_vars; ++i) {
    x.push_back(p.var(-30, 30));
  }
  std::vector<Expr<double>> terms;
  terms.reserve(num_vars - 1);
  for (int i = 0; i < num_vars - 1; ++i) {
    auto a = x[i] * x[i] - x[i+1];
    auto b = x[i] - 1;
    terms.push_back(a * a * 100 + b * b);
  }
  Expr<double> obj = terms[0];
  for (std::size_t i = 1; i < terms.size(); ++i) {
    obj = obj + terms[i];
  }
  p.set_objective(obj);
  return p;
}

namespace {
constexpr std::size_t DIMS = 100;
constexpr std::size_t CYCLE_SIZE = 5;
constexpr std::size_t PARTITION_NUM = 4;
constexpr std::size_t SAMPLE_POINTS = 10;
constexpr std::size_t NUM_CHILDREN = rosenbrock::ipow<PARTITION_NUM, CYCLE_SIZE>();
}


class graph_driver : public driver {
public:
  auto solve() -> double {
    using search::CompressedInterval;
    using search::IntervalHistory;
    using search::IntervalPQueue;

    Problem<double> rosenbrock_problem = create_rosenbrock(DIMS);

    // Built once, replayed every iteration against a new box/cycle_start
    auto point_replay = PointGraphReplay<double, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS>::build(
        rosenbrock_problem, NUM_CHILDREN);
    auto interval_replay =
        IntervalGraphReplay<double, CYCLE_SIZE, PARTITION_NUM>::build(rosenbrock_problem, NUM_CHILDREN);

    IntervalPQueue<double> pending(1000);
    IntervalHistory<double> history;

    std::vector<cu::interval<double>> origin {};
    history.enqueue(origin);

    pending.enqueue(CompressedInterval<double> {
      .sidx = 0,
      .pidx = 0,
      .cycle_start = 0,
      .lb = GLB_,
    });

    std::vector<double> child_lb_storage(NUM_CHILDREN);
    std::span<double, NUM_CHILDREN> child_lb(child_lb_storage.data(), NUM_CHILDREN);

    bool converged = false;

    while (iter_idx_ < iter_limit_ && !pending.empty() && GUB_ - GLB_ > tolerance_)
    {
      ++iter_idx_;

      CompressedInterval<double> cur = pending.dequeue();

      if (cur.lb > GUB_) {
        converged = true;
        break;
      }

      GLB_ = cur.lb;

      std::vector<cu::interval<double>> box;
      box.resize(DIMS);
      if (cur.pidx == 0) {
        // TODO: this is for the "no parent" fallback, but each 
        // variable in problem will have box bounds now
        box = rosenbrock_problem.box_bounds;
      } else {
        cur.materialise(history, box, CYCLE_SIZE, PARTITION_NUM);
      }

      std::size_t const box_idx = history.enqueue(box);
      std::size_t const child_cycle_start = (cur.cycle_start + CYCLE_SIZE) % DIMS;

      // GUB sampling: candidate is the best sampled, feasible value from this domain
      point_replay.set_domain(box, child_cycle_start);
      point_replay.launch(/*stream=*/0);
      GUB_ = std::min(GUB_, point_replay.candidate());

      // Interval analysis of sub-domains and GUB pruning
      interval_replay.set_domain(box, child_cycle_start);
      interval_replay.launch(/*stream=*/0);
      auto obj_lb = interval_replay.obj_lb();
      for (std::size_t tid = 0; tid < NUM_CHILDREN; ++tid) {
        child_lb[tid] = obj_lb[tid];
        if (obj_lb[tid] > GUB_) continue;

        pending.enqueue(CompressedInterval<double> {
            .sidx = tid,
            .pidx = box_idx,
            .cycle_start = child_cycle_start,
            .lb = obj_lb[tid],
        });
      }

      std::cout << "iter " << iter_idx_ << ": GUB = " << GUB_ << '\n';
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

auto main() -> int
{
  graph_driver drv;
  drv.solve();

  return 0;
}

}