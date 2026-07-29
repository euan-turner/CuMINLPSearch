#include <iostream>
#include <string>

#include "cuminlp/dag.hpp"
#include "cuminlp/graph_driver.cuh"

using cuminlp::dag::Expr;
using cuminlp::dag::Problem;
using cuminlp::dag::Cmp;

namespace
{
// https://www.minlplib.org/ex2_1_2.html
Problem<double> problem_212() {
  std::vector<Expr<double>> x;
  Problem<double> problem;
  x.reserve(6);
  for (int i = 0; i < 5; ++i) {
    x.push_back(problem.var(0, 1));
  }
  x.push_back(problem.var(0, 20));  // no explicit upper bound; 20 is implied by c2 since x1,x3 >= 0
  Expr<double> obj = -0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + x[4] * x[4]);
  obj = obj - 10.5 * x[0] - 7.5 * x[1] - 3.5 * x[2] - 2.5 * x[3] - 1.5 * x[4] - 10 * x[5];
  problem.set_objective(obj);

  Expr<double> c1 = 6 * x[0] + 3 * x[1] + 3 * x[2] + 2 * x[3] + x[4];
  problem.add_constraint(c1, Cmp::LE, 6.5);
  Expr<double> c2 = 10 * x[0] + 10 * x[3] + x[5];
  problem.add_constraint(c2, Cmp::LE, 20);
  return problem;
}

// https://www.minlplib.org/ex2_1_9.html
Problem<double> problem_219() {
  Problem<double> problem;
  auto x1 = problem.var(0, 1);
  auto x2 = problem.var(0, 1);
  auto x3 = problem.var(0, 1);
  auto x4 = problem.var(0, 1);
  auto x5 = problem.var(0, 1);
  auto x6 = problem.var(0, 1);
  auto x7 = problem.var(0, 1);
  auto x8 = problem.var(0, 1);
  auto x9 = problem.var(0, 1);
  auto x10 = problem.var(0, 1);

  Expr<double> obj = x1*x2 + x2*x3 + x3*x4 + x4*x5 + x5*x6 + x6*x7 + x7*x8 + x8*x9 + x9*x10 + x1*x3 + x2*x4 + x3*x5 + x4*x6 + x5*x7 + x6*x8 + x7*x9 + x8*x10 + x1*x9 + x1*x10 + x2*x10 + x1*x5 + x4*x7;
  problem.set_objective(-obj);

  Expr<double> c1 = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10;
  problem.add_constraint(c1, Cmp::EQ, 1);
  return problem;
}
} 

auto main(int argc, char* argv[]) -> int
{
  int problem = std::stoi(argv[1]);
  int iters = std::stoi(argv[2]);
  if (problem == 212) {
    std::cout << "Ex. 2.1.2" << '\n';
    cuminlp::GraphDriver<double, 2, 20, 100> drv_212(iters, 1e-9);
    drv_212.solve(problem_212());
  } else if (problem == 219) {
    std::cout << "Ex. 2.1.9" << '\n';
    cuminlp::GraphDriver<double, 2, 20, 100> drv_219(iters, 1e-9);
    drv_219.solve(problem_219());
  }
  return 0;
}