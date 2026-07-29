// https://www.minlplib.org/autocorr_bern20-03.html

#include <memory>
#include <string>
#include <vector>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/graph_driver.cuh"

using cuminlp::dag::Expr;
using cuminlp::dag::Problem;

namespace {

constexpr std::size_t NUM_VARS = 20;
constexpr std::size_t CYCLE_SIZE = 20;
constexpr std::size_t PARTITION_NUM = 2;  // every variable is binary
constexpr std::size_t SAMPLE_POINTS = 1;  // binary domain has no interior to sample

Problem<double> make_autocorr_bern20_03()
{
  Problem<double> problem;
  std::vector<Expr<double>> b;
  b.reserve(NUM_VARS);
  for (std::size_t j = 0; j < NUM_VARS; ++j) {
    b.push_back(problem.bin_var());
  }

  Expr<double> obj = 8.0 * (b[0] * b[2]);
  for (std::size_t i = 1; i < NUM_VARS - 2; ++i) {
    obj = obj + 8.0 * (b[i] * b[i + 2]);
  }

  // b1, b2, b19, b20 (indices 0, 1, 18, 19) each appear in exactly one
  // b_i*b_{i+2} pair, so their linear coefficient is 4; b3..b18 each appear
  // in two pairs (once as b_i, once as b_{i+2}), so theirs is 8.
  auto linear_coeff = [](std::size_t j) -> double {
    return (j == 0 || j == 1 || j == NUM_VARS - 2 || j == NUM_VARS - 1) ? 4.0 : 8.0;
  };
  for (std::size_t j = 0; j < NUM_VARS; ++j) {
    obj = obj - linear_coeff(j) * b[j];
  }

  problem.set_objective(obj);
  return problem;
}

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  int iters = argc > 1 ? std::stoi(argv[1]) : 10;

  auto policy =
      std::make_shared<cuminlp::GreedyCompositionPolicy<double, CYCLE_SIZE, PARTITION_NUM>>();
  cuminlp::GraphDriver<double, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS> drv(policy, iters, 1e-6);
  drv.solve(make_autocorr_bern20_03());

  return 0;
}
