#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "cuminlp/model/problem.hpp"

namespace cuminlp::testing
{

/**
 * @brief Host-side real-valued evaluator for an ExprDAG.
 *
 * Plain `T` arithmetic, no interval bounding: this is the reference oracle the
 * frontend tests compare against, and (later) the CPU counterpart for checking
 * that the GPU kernels' directed rounding encloses the true value.
 *
 * Node ids are topologically ordered (every operand id is smaller), so one
 * forward sweep up to `root` evaluates everything `root` depends on. Nodes
 * belonging to other functions are evaluated too; that is wasted work but
 * never wrong, and keeps this to a single loop.
 */
template<typename T>
auto evaluate(model::ExprDAG<T> const& graph,
              std::size_t root,
              std::vector<T> const& point) -> T
{
  using model::Op;

  if (root >= graph.nodes.size()) {
    throw std::runtime_error("evaluate: root id out of range");
  }

  std::vector<T> value(root + 1, T {});

  for (std::size_t i = 0; i <= root; ++i) {
    auto const& node = graph.nodes[i];
    auto arg = [&](std::size_t k) { return value[node.in[k]]; };

    switch (node.op) {
      case Op::Var:
        if (node.payload.var_index >= point.size()) {
          throw std::runtime_error("evaluate: point has too few components");
        }
        value[i] = point[node.payload.var_index];
        break;
      case Op::Const:
        value[i] = node.payload.constant;
        break;
      case Op::Add:
        value[i] = arg(0) + arg(1);
        break;
      case Op::Sub:
        value[i] = arg(0) - arg(1);
        break;
      case Op::Mul:
        value[i] = arg(0) * arg(1);
        break;
      case Op::Div:
        value[i] = arg(0) / arg(1);
        break;
      case Op::Min:
        value[i] = std::fmin(arg(0), arg(1));
        break;
      case Op::Max:
        value[i] = std::fmax(arg(0), arg(1));
        break;
      case Op::Neg:
        value[i] = -arg(0);
        break;
      case Op::Sqr:
        value[i] = arg(0) * arg(0);
        break;
      case Op::Exp:
        value[i] = std::exp(arg(0));
        break;
      case Op::Log:
        value[i] = std::log(arg(0));
        break;
      case Op::Sqrt:
        value[i] = std::sqrt(arg(0));
        break;
      case Op::Sin:
        value[i] = std::sin(arg(0));
        break;
      case Op::Cos:
        value[i] = std::cos(arg(0));
        break;
      case Op::Tanh:
        value[i] = std::tanh(arg(0));
        break;
      case Op::Abs:
        value[i] = std::fabs(arg(0));
        break;
      case Op::PowN:
        value[i] = std::pow(arg(0), node.payload.int_exp);
        break;
      case Op::Pow:
        value[i] = std::pow(arg(0), arg(1));
        break;
      default:
        throw std::runtime_error("evaluate: unhandled Op");
    }
  }

  return value[root];
}

/// Convenience overload: evaluate a Problem's objective.
template<typename T>
auto evaluate_objective(model::Problem<T> const& problem,
                        std::vector<T> const& point) -> T
{
  return evaluate(problem.graph, problem.objective_root, point);
}

}  // namespace cuminlp::testing
