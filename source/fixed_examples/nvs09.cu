// https://www.minlplib.org/nvs09.html

#include <memory>
#include <string>

#include "cuminlp/backend/graph/factory.cuh"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/example_main.hpp"
#include "cuminlp/policy/greedy_enum.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/driver.hpp"
#include "nvs09_problem.hpp"

using namespace cuminlp::examples::nvs09;

auto main(int argc, char* argv[]) -> int
{
  return cuminlp::examples::guarded(
      [&]
      {
        int iters = argc > 1 ? std::stoi(argv[1]) : 2000;

        auto problem = make_nvs09();

        auto policy =
            std::make_shared<cuminlp::policy::GreedyEnumCompositionPolicy<double>>(
                cuminlp::region::FanOutSpec {PARTITION_NUM},
                cuminlp::config::SearchCalibration {.max_cycle_size =
                                                        CYCLE_SIZE});
        auto backend = std::make_shared<
            const cuminlp::backend::graph::GraphBackendFactory<double>>();
        auto reporter = std::make_shared<cuminlp::report::ConsoleReporter>(
            cuminlp::config::profile_problem(problem));
        cuminlp::search::SearchDriver<double> drv(
            policy,
            backend,
            iters,
            1e-6,
            SAMPLE_POINTS,
            0,
            0,
            cuminlp::search::FrontierPolicy::StopAtBudget,
            false,
            reporter);
        drv.solve(problem);
      });
}
