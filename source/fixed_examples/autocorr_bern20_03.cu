// https://www.minlplib.org/autocorr_bern20-03.html

#include <memory>
#include <string>

#include "autocorr_bern20_03_problem.hpp"
#include "cuminlp/backend/graph/factory.cuh"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/example_main.hpp"
#include "cuminlp/policy/greedy_enum.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/driver.hpp"

using namespace cuminlp::examples::autocorr_bern20_03;

auto main(int argc, char* argv[]) -> int
{
  return cuminlp::examples::guarded(
      [&]
      {
        int iters = argc > 1 ? std::stoi(argv[1]) : 10;

        auto problem = make_autocorr_bern20_03();

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
            reporter);
        drv.solve(problem);
      });
}
