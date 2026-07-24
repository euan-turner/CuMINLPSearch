#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "cuminlp/cuminlp.hpp"
#include "cuminlp/search.hpp"
#include "cuminlp/rosenbrock.cuh"

namespace cuminlp
{

driver::driver(uint32_t iter_limit, double tolerance)
    : GUB_(std::numeric_limits<double>::max())
    , GLB_(std::numeric_limits<double>::lowest())
    , tolerance_(tolerance)
    , iter_limit_(iter_limit)
    , iter_idx_(0)
{
}

namespace
{
// Rosenbrock instance and partitioning scheme the driver searches.
// TODO: accept a general QCQP instance/parameters instead of hardcoding
// Rosenbrock over [-30, 30]^DIMS with a fixed cycling scheme.
// TODO: implicit invariant: DIMS % CYCLE_SIZE == 0

constexpr std::size_t DIMS = 100;
constexpr std::size_t CYCLE_SIZE = 5;
constexpr std::size_t PARTITION_NUM = 4;
constexpr std::size_t SAMPLE_POINTS = 10;
constexpr std::size_t NUM_CHILDREN =
    rosenbrock::ipow<PARTITION_NUM, CYCLE_SIZE>();
}  // namespace

auto driver::solve() -> double
{
  using search::CompressedInterval;
  using search::IntervalHistory;
  using search::IntervalPQueue;

  // Initialise the pending list on CPU with a single (whole-domain) interval
  IntervalPQueue<double> pending(1000);
  IntervalHistory<double> history;

  // Index 0 is reserved for the original index, special-cased by
  // CompressedInterval::materialise
  std::vector<cu::interval<double>> origin {};
  history.enqueue(origin);

  pending.enqueue(CompressedInterval<double> {
      .sidx = 0,
      .pidx = 0,
      .cycle_start = 0,
      .lb = GLB_,
  });

  // NUM_CHILDREN = PARTITION_NUM^CYCLE_SIZE can be far too large to hold as
  // stack-local std::array (e.g. 4^10 ~= 1M), so these live on the heap and
  // are reused across iterations.
  auto pruned_storage = std::make_unique<bool[]>(NUM_CHILDREN);
  std::vector<double> child_lb_storage(NUM_CHILDREN);
  std::span<bool, NUM_CHILDREN> pruned(pruned_storage.get(), NUM_CHILDREN);
  std::span<double, NUM_CHILDREN> child_lb(child_lb_storage.data(),
                                           NUM_CHILDREN);

  // set once the pending set is proven fully dominated by GUB_, either by
  // draining it or by the lazy-pruning break below
  bool converged = false;

  // convergence condition is tightness of region, iteration limit, or
  // completion
  while (iter_idx_ < iter_limit_ && !pending.empty()
         && GUB_ - GLB_ > tolerance_)
  {
    ++iter_idx_;

    // Find region with least lower bound
    CompressedInterval<double> cur = pending.dequeue();

    // lazy pruning: cur holds the least lb of every region still pending
    // (min-heap invariant), so if it's already dominated by GUB_ then every
    // other pending region is too -- the search is fully converged, no need
    // to drain the rest of the heap one skipped pop at a time.
    if (cur.lb > GUB_) {
      converged = true;
      break;
    }

    // pending is a min-heap on lb, so the region just popped holds the
    // tightest lower bound over everything still outstanding.
    GLB_ = cur.lb;

    std::vector<cu::interval<double>> box;
    box.resize(DIMS);
    if (cur.pidx == 0) {
      // Root: materialise()'s generic "no parent" fallback is an unbounded
      // box, but this problem's actual domain is [-30, 30]^DIMS (Rosenbrock's
      // constraint). TODO: drop this once the driver accepts a real problem
      // definition instead of hardcoding Rosenbrock.
      for (auto& b : box) {
        b = cu::interval<double>(-30.0, 30.0);
      }
    } else {
      cur.materialise(history, box, CYCLE_SIZE, PARTITION_NUM);
    }

    // register this region's own explicit box so its children (created
    // below) can look it up as their parent.
    std::size_t const box_idx = history.enqueue(box);

    // Cycle to the next block of dimensions to partition.
    std::size_t const child_cycle_start = (cur.cycle_start + CYCLE_SIZE) % DIMS;

    // update GUB from sampled points in the current region
    double const lub = rosenbrock::launch_sample_rosenbrock<double,
                                                            CYCLE_SIZE,
                                                            PARTITION_NUM,
                                                            SAMPLE_POINTS>(
        box, child_cycle_start);
    GUB_ = std::min(GUB_, lub);

    // interval analysis over subregions
    rosenbrock::launch_bound_rosenbrock<double, CYCLE_SIZE, PARTITION_NUM>(
        box, GUB_, child_cycle_start, pruned, child_lb);

    // CPU updates GUB and prunes newsubregions that are suboptimal
    for (std::size_t tid = 0; tid < NUM_CHILDREN; ++tid) {
      if (pruned[tid]) {
        continue;
      }
      pending.enqueue(CompressedInterval<double> {
          .sidx = tid,
          .pidx = box_idx,
          .cycle_start = child_cycle_start,
          .lb = child_lb[tid],
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
  // TODO: interval hull of all viable regions
  return GUB_;
}

}  // namespace cuminlp
