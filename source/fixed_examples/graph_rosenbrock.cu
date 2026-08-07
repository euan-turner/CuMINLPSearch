// https://www.minlplib.org/prob09.html, but with more than 2 dimensions

#include <cstddef>
#include <memory>
#include <vector>

#include "cuminlp/backend/graph/factory.cuh"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/example_main.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/greedy_enum.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/driver.hpp"

using cuminlp::model::Expr;
using cuminlp::model::Problem;

namespace
{

constexpr std::size_t DIMS = 100;
constexpr std::size_t CYCLE_SIZE = 5;
constexpr std::size_t PARTITION_NUM = 4;
constexpr std::size_t SAMPLE_POINTS = 10;

Problem<double> make_rosenbrock(std::size_t num_vars)
{
  Problem<double> p;
  std::vector<Expr<double>> x;
  x.reserve(num_vars);
  for (std::size_t i = 0; i < num_vars; ++i) {
    x.push_back(p.var(-30, 30));
  }
  std::vector<Expr<double>> terms;
  terms.reserve(num_vars - 1);
  for (std::size_t i = 0; i < num_vars - 1; ++i) {
    auto a = x[i] * x[i] - x[i + 1];
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

}  // namespace

auto main() -> int
{
  return cuminlp::examples::guarded(
      [&]
      {
        Problem<double> problem = make_rosenbrock(DIMS);

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
            1000000,
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
