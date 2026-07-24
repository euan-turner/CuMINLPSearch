// Builds the same N-dimensional Rosenbrock instance as source/main.cpp
// (cuminlp::rosenbrock::FixedRosenbrockDriver), but as a dag::Problem, and
// solves it via cuminlp::GraphDriver (include/cuminlp/graph_driver.cuh),
// which evaluates each region through CUDA-graph replays built from the
// problem's expression DAG rather than the hand-written per-op kernels in
// include/cuminlp/rosenbrock.cuh. Intended to be run side-by-side with
// cuminlp_exe to compare the two evaluators.

#include <cstddef>
#include <vector>

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

  cuminlp::GraphDriver<double, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS> drv;
  drv.solve(problem);

  return 0;
}
