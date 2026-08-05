// https://www.minlplib.org/autocorr_bern20-03.html

#include <memory>
#include <string>

#include "autocorr_bern20_03_problem.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/example_main.hpp"
#include "cuminlp/graph_driver.cuh"

using namespace cuminlp::examples::autocorr_bern20_03;

auto main(int argc, char* argv[]) -> int
{
  return cuminlp::examples::guarded(
      [&]
      {
        int iters = argc > 1 ? std::stoi(argv[1]) : 10;

        auto policy = std::make_shared<cuminlp::GreedyCompositionPolicy<double>>(
            cuminlp::FanOutSpec {PARTITION_NUM},
            cuminlp::SearchCalibration {.max_cycle_size = CYCLE_SIZE});
        cuminlp::GraphDriver<double> drv(policy, iters, 1e-6, SAMPLE_POINTS);
        drv.solve(make_autocorr_bern20_03());
      });
}
