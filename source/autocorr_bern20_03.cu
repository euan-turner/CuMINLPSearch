// https://www.minlplib.org/autocorr_bern20-03.html

#include <memory>
#include <string>

#include "autocorr_bern20_03_problem.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/graph_driver.cuh"

using namespace cuminlp::examples::autocorr_bern20_03;

auto main(int argc, char* argv[]) -> int
{
  int iters = argc > 1 ? std::stoi(argv[1]) : 10;

  auto policy = std::make_shared<
      cuminlp::GreedyCompositionPolicy<double, CYCLE_SIZE, PARTITION_NUM>>();
  cuminlp::GraphDriver<double, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS> drv(
      policy, iters, 1e-6);
  drv.solve(make_autocorr_bern20_03());

  return 0;
}
