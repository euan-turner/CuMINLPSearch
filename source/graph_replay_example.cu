// Problem: minimise f(x, y) = x^2 + 2y over x in [-1, 1], y in [0, 5],
// subject to x + y <= 5. CycleSize=2, PartitionNum=4 partitions both
// variables into 4 slices each, giving 4^2 = 16 regions.

#include <cstdio>
#include <vector>

#include "cuminlp/dag.hpp"
#include "cuminlp/graph_replay.cuh"

using namespace cuminlp::dag;

namespace {

constexpr std::size_t CYCLE_SIZE = 2;
constexpr std::size_t PARTITION_NUM = 4;
constexpr std::size_t NUM_REGIONS = 16;  // PARTITION_NUM ^ CYCLE_SIZE

// Mirrors partition::get_bounds/make_cycle_context (partition.cuh) on the
// host, purely to print each region's box alongside its computed bounds --
// not part of the evaluator itself.
cu::interval<double> region_bound(cu::interval<double> parent, std::size_t region,
                                  std::size_t dim)
{
  std::size_t part = (dim == 0) ? region % PARTITION_NUM : (region / PARTITION_NUM) % PARTITION_NUM;
  double width = (parent.ub - parent.lb) / PARTITION_NUM;
  return cu::interval<double>(parent.lb + width * part, parent.lb + width * (part + 1));
}

}  // namespace

auto main() -> int
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  auto y = p.var(0.0, 5.0);
  auto objective = x * x + 2.0 * y;
  p.set_objective(objective);
  p.add_constraint(x + y, Cmp::LE, 5.0);

  auto replay = IntervalGraphReplay<double, CYCLE_SIZE, PARTITION_NUM>::build(p, NUM_REGIONS);

  std::vector<cu::interval<double>> domain = {
      cu::interval<double>(-1.0, 1.0), cu::interval<double>(0.0, 5.0)};
  replay.set_domain(domain, /*cycle_start=*/0);
  replay.launch(/*stream=*/0);

  auto feasible = replay.feasible();
  auto obj_lb = replay.obj_lb();

  std::printf("%-6s %-16s %-16s %-9s %10s\n", "region", "x range", "y range", "feasible",
             "obj_lb");
  for (std::size_t r = 0; r < NUM_REGIONS; ++r) {
    cu::interval<double> xr = region_bound(domain[0], r, 0);
    cu::interval<double> yr = region_bound(domain[1], r, 1);
    std::printf("%-6zu [%5.2f,%5.2f] [%5.2f,%5.2f] %-9d %10.4f\n", r, xr.lb, xr.ub, yr.lb,
               yr.ub, static_cast<int>(feasible[r]), obj_lb[r]);
  }

  return 0;
}
