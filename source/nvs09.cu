// https://www.minlplib.org/nvs09.html

#include <memory>
#include <string>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/graph_driver.cuh"
#include "nvs09_problem.hpp"

using namespace cuminlp::examples::nvs09;

auto main(int argc, char* argv[]) -> int
{
  int iters = argc > 1 ? std::stoi(argv[1]) : 2000;

  auto policy = std::make_shared<
      cuminlp::GreedyCompositionPolicy<double, CYCLE_SIZE, PARTITION_NUM>>();
  cuminlp::GraphDriver<double, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS> drv(
      policy, iters, 1e-6);
  drv.solve(make_nvs09());

  return 0;
}
