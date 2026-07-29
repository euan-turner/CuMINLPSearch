// https://www.minlplib.org/nvs09.html

#include <memory>
#include <string>
#include <vector>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/graph_driver.cuh"

using cuminlp::dag::Expr;
using cuminlp::dag::Problem;

namespace
{

constexpr std::size_t NUM_VARS = 10;
constexpr std::size_t CYCLE_SIZE = 10;
constexpr std::size_t PARTITION_NUM = 7;  // every integer variable is [3, 9]
constexpr std::size_t SAMPLE_POINTS = 5;

Problem<double> make_nvs09()
{
  Problem<double> problem;
  std::vector<Expr<double>> i;
  i.reserve(NUM_VARS);
  for (std::size_t j = 0; j < NUM_VARS; ++j) {
    i.push_back(problem.int_var(3.0, 9.0));
  }

  Expr<double> product = i[0];
  for (std::size_t j = 1; j < NUM_VARS; ++j) {
    product = product * i[j];
  }

  Expr<double> sum_terms = sqr(log(i[0] - 2.0)) + sqr(log(10.0 - i[0]));
  for (std::size_t j = 1; j < NUM_VARS; ++j) {
    sum_terms = sum_terms + sqr(log(i[j] - 2.0)) + sqr(log(10.0 - i[j]));
  }

  problem.set_objective(sum_terms - exp(0.2 * log(product)));
  return problem;
}

}  // namespace

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
