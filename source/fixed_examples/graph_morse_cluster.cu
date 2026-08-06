// https://www.minlplib.org/ex8_6_2.html
// complex objective, no constraints

#include <memory>
#include <string>

#include "cuminlp/backend/graph/factory.cuh"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/example_main.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/greedy.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/driver.hpp"

using cuminlp::model::Expr;
using cuminlp::model::Problem;

namespace
{

// 30 variables, 6 fixed to break translational/rotational symmetry -> 24 free
// dims
constexpr std::size_t POINTS = 10;
constexpr std::size_t VARS = 3 * POINTS;
constexpr std::size_t CYCLE_SIZE = 2;
constexpr std::size_t PARTITION_NUM = 20;
constexpr std::size_t SAMPLE_POINTS = 10;
}  // namespace

// 10000 iterations
// subdomains is CYCLE_SIZE^PARTITION_NUM
// CYCLE 4, PARTITION 10 = -22.0167
// CYCLE 2, PARTITION 20 = -16.2154
// CYCLE 8, PARTITION 5 =
auto main(int argc, char* argv[]) -> int
{
  return cuminlp::examples::guarded(
      [&]
      {
        Problem<double> problem;
        std::vector<Expr<double>> x;
        x.reserve(VARS);
        // x1, x11, x12, x21, x22, x23 are all fixed to 0 to break translational
        // symmetry
        for (std::size_t i = 0; i < VARS; ++i) {
          if (i == 0 || i == 10 || i == 11 || i == 20 || i == 21 || i == 22) {
            x.push_back(problem.fixed(0));
          } else {
            x.push_back(problem.var(-5, 5));
          }
        }

        std::vector<Expr<double>> terms;
        terms.reserve(POINTS * (POINTS - 1) / 2);

        for (std::size_t i = 0; i < POINTS; ++i) {
          for (std::size_t j = i + 1; j < POINTS; ++j) {
            auto dx = x[i] - x[j];
            auto dy = x[i + 10] - x[j + 10];
            auto dz = x[i + 20] - x[j + 20];
            auto r2 = (dx * dx) + (dy * dy) + (dz * dz);
            auto r = sqrt(r2);
            auto t = exp(3 * (1 - r));
            auto u = 1 - t;
            auto v = u * u;
            terms.push_back(v);
          }
        }

        Expr<double> obj = terms[0];
        for (std::size_t i = 1; i < terms.size(); ++i) {
          obj = obj + terms[i];
        }
        problem.set_objective(-45 + obj);

        int iters = std::stoi(argv[1]);

        auto policy =
            std::make_shared<cuminlp::policy::GreedyCompositionPolicy<double>>(
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
