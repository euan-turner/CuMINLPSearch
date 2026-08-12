#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "cuminlp/model/problem.hpp"

// Host-side real-valued evaluation of an ExprDAG: the reference oracle the
// frontend tests compare a parsed Problem against, and the ground truth the
// aggregate backend's bounds are checked to enclose
// (design/AGGREGATE_BOUNDING.md §11, tests 7-8).
//
// Promoted here from test/source/dag_eval.hpp, which put it out of reach of
// anything outside the test tree (design/AGGREGATE_BOUNDING.md §12, stage 1).
//
// Deliberately host-only and device-free, for the same reason
// Problem::validate() is: it names model/problem.hpp and <cmath> and nothing
// else, so every target that can parse a Problem can also evaluate one,
// with or without a CUDA toolchain.
//
// Plain `T` arithmetic with no directed rounding. That is what makes it a
// useful independent check on the interval kernels -- two implementations
// that shared a rounding mode would agree about a bug in it -- and equally
// what makes it unfit to *be* a bound. Nothing here returns anything sound.
namespace cuminlp::model
{

namespace detail
{

/// Values of nodes `0..last` inclusive, in one forward sweep. Node ids are
/// topologically ordered (every operand id is strictly smaller, an invariant
/// Problem::validate() checks), so one pass suffices. Precondition:
/// `last < graph.nodes.size()`.
template<typename T>
auto evaluate_prefix(ExprDAG<T> const& graph,
                     std::size_t last,
                     std::vector<T> const& point) -> std::vector<T>
{
  std::vector<T> value(last + 1, T {});

  for (std::size_t i = 0; i <= last; ++i) {
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

  return value;
}

}  // namespace detail

/**
 * @brief Every node's value at `point`, in one forward sweep.
 *
 * Returned whole rather than per-root because a Problem has one objective
 * root and one root per constraint: evaluating each separately re-sweeps the
 * shared subgraph once per root, which is quadratic in a graph whose
 * constraints share most of their structure.
 *
 * @throws std::runtime_error if `point` has fewer components than the graph
 *         has variables, or if a node carries an Op this function does not
 *         handle.
 */
template<typename T>
auto evaluate_all(ExprDAG<T> const& graph,
                  std::vector<T> const& point) -> std::vector<T>
{
  if (graph.nodes.empty()) {
    return {};
  }
  return detail::evaluate_prefix(graph, graph.nodes.size() - 1, point);
}

/**
 * @brief The value of the expression rooted at `root`, at `point`.
 *
 * Sweeps only as far as `root`. Nodes belonging to other functions below it
 * are evaluated too; that is wasted work but never wrong, and keeps this to
 * a single loop.
 *
 * @throws std::runtime_error if `root` is out of range, `point` has too few
 *         components, or a node carries an unhandled Op.
 */
template<typename T>
auto evaluate(ExprDAG<T> const& graph,
              std::size_t root,
              std::vector<T> const& point) -> T
{
  if (root >= graph.nodes.size()) {
    throw std::runtime_error("evaluate: root id out of range");
  }
  return detail::evaluate_prefix(graph, root, point)[root];
}

/// Convenience overload: evaluate a Problem's objective.
template<typename T>
auto evaluate_objective(Problem<T> const& problem,
                        std::vector<T> const& point) -> T
{
  return evaluate(problem.graph, problem.objective_root, point);
}

/**
 * @brief Whether `lhs` satisfies `cmp rhs`, to the device's own tolerances.
 *
 * Mirrors `backend::graph::is_feasible(T, Cmp, T)` and its `almost_equal`
 * (backend/graph/kernels.cuh): `LE` is exact, `EQ` is an absolute-or-relative
 * tolerance. It must keep mirroring it -- this is the host statement of the
 * predicate the device sampler applies to every drawn point, and the two
 * disagreeing would show up as a "feasible" point the search rejects or an
 * infeasible one it accepts.
 *
 * A NaN `lhs` (from, say, log of a negative operand) fails every comparison
 * and is therefore reported infeasible, which is the safe direction and is
 * what the device does too.
 */
template<typename T>
auto satisfies(T lhs, Cmp cmp, T rhs, T abs_tol = T(1e-8), T rel_tol = T(1e-6))
    -> bool
{
  if (cmp == Cmp::LE) {
    return lhs <= rhs;
  }
  if (cmp == Cmp::EQ) {
    T const diff = std::fabs(lhs - rhs);
    if (diff <= abs_tol) {
      return true;
    }
    return diff <= rel_tol * std::fmax(std::fabs(lhs), std::fabs(rhs));
  }
  return false;
}

/**
 * @brief Index of the first constraint `point` violates, if any.
 *
 * Returns the index rather than a bare bool so a failing check can say which
 * constraint failed -- for a 200-constraint model, "infeasible" on its own is
 * not something you can act on.
 *
 * Constraints only. Box containment and integrality are
 * `search::witness_is_admissible`'s (search/epilogue.hpp), and are
 * deliberately not duplicated here.
 *
 * This is a reference oracle for tests and diagnostics, **not** the search's
 * feasibility check: on the incumbent path that stays the device's job
 * (`feasibility_check_kernel`), for the reason witness_is_admissible gives --
 * a second host-side implementation is a second thing to disagree with. The
 * aggregate backend does not change that (design/AGGREGATE_BOUNDING.md §6.2).
 */
template<typename T>
auto first_violated_constraint(Problem<T> const& problem,
                               std::vector<T> const& point,
                               T abs_tol = T(1e-8),
                               T rel_tol = T(1e-6))
    -> std::optional<std::size_t>
{
  std::vector<T> const value = evaluate_all(problem.graph, point);
  for (std::size_t i = 0; i < problem.constraints.size(); ++i) {
    ConstraintRef<T> const& c = problem.constraints[i];
    if (c.root_id >= value.size()) {
      throw std::runtime_error(
          "first_violated_constraint: constraint root id out of range");
    }
    if (!satisfies(value[c.root_id], c.cmp, c.rhs, abs_tol, rel_tol)) {
      return i;
    }
  }
  return std::nullopt;
}

/// True iff `point` satisfies every constraint. See
/// `first_violated_constraint` for what this does and does not cover.
template<typename T>
auto satisfies_constraints(Problem<T> const& problem,
                           std::vector<T> const& point,
                           T abs_tol = T(1e-8),
                           T rel_tol = T(1e-6)) -> bool
{
  return !first_violated_constraint(problem, point, abs_tol, rel_tol)
              .has_value();
}

}  // namespace cuminlp::model
