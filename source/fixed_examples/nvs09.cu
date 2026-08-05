// https://www.minlplib.org/nvs09.html

#include <memory>
#include <string>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/example_main.hpp"
#include "cuminlp/graph_driver.cuh"
#include "nvs09_problem.hpp"

using namespace cuminlp::examples::nvs09;

auto main(int argc, char* argv[]) -> int
{
  return cuminlp::examples::guarded(
      [&]
      {
        int iters = argc > 1 ? std::stoi(argv[1]) : 2000;

        auto policy = std::make_shared<cuminlp::GreedyCompositionPolicy<double>>(
            cuminlp::FanOutSpec {PARTITION_NUM},
            cuminlp::SearchCalibration {.max_cycle_size = CYCLE_SIZE});
        cuminlp::GraphDriver<double> drv(policy, iters, 1e-6, SAMPLE_POINTS);
        drv.solve(make_nvs09());
      });
}
