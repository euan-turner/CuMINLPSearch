// https://www.minlplib.org/prob09.html, but with more than 2 dimensions

#include <cstddef>
#include <memory>
#include <vector>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/graph_driver.cuh"

using cuminlp::dag::Expr;
using cuminlp::dag::Problem;

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
  Problem<double> problem = make_rosenbrock(DIMS);

  auto policy =
      std::make_shared<cuminlp::GreedyCompositionPolicy<double, CYCLE_SIZE, PARTITION_NUM>>();
  cuminlp::GraphDriver<double, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS> drv(policy);
  drv.solve(problem);

  return 0;
}
