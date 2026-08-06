#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/errors.hpp"
#include "cuminlp/model/dag.hpp"

// The MINLP problem wrapper built on model/dag.hpp's expression IR
namespace cuminlp::model
{

enum class VarKind
{
  Continuous,
  Integer,
  Binary
};
enum class Cmp
{
  LE,
  EQ
};

/**
 * @brief Expression in the expression DAG, corresponding to variable, constant
 * or operation.
 *
 * @tparam T
 */
template<typename T>
class Expr
{
public:
  Expr(ExprDAG<T>* g, std::size_t id)
      : graph(g)
      , node_id(id)
  {
  }

  friend Expr operator+(Expr a, Expr b)
  {
    return {a.graph, a.graph->emit(Op::Add, {a.node_id, b.node_id})};
  }

  friend Expr operator-(Expr a, Expr b)
  {
    return {a.graph, a.graph->emit(Op::Sub, {a.node_id, b.node_id})};
  }

  friend Expr operator-(Expr a)
  {
    if (a.is_const()) {
      return a.constant(-a.const_value());
    }
    return {a.graph, a.graph->emit(Op::Neg, {a.node_id})};
  }

  friend Expr operator*(Expr a, Expr b)
  {
    if (a.node_id == b.node_id) {
      if (a.is_const()) {
        return a.constant(a.const_value() * a.const_value());
      }
      return {a.graph, a.graph->emit(Op::Sqr, {a.node_id})};
    } else {
      return {a.graph, a.graph->emit(Op::Mul, {a.node_id, b.node_id})};
    }
  }

  friend Expr operator/(Expr a, Expr b)
  {
    return {a.graph, a.graph->emit(Op::Div, {a.node_id, b.node_id})};
  }

  friend Expr operator+(Expr a, T b) { return a + a.constant(b); }

  friend Expr operator+(T a, Expr b) { return b.constant(a) + b; }

  friend Expr operator-(Expr a, T b) { return a - a.constant(b); }

  friend Expr operator-(T a, Expr b) { return b.constant(a) - b; }

  friend Expr operator*(Expr a, T b) { return a * a.constant(b); }

  friend Expr operator*(T a, Expr b) { return b.constant(a) * b; }

  friend Expr operator/(Expr a, T b) { return a / a.constant(b); }

  friend Expr operator/(T a, Expr b) { return b.constant(a) / b; }

  // Unary ops applied directly to a Const operand constant-fold transparently
  // rather than emitting a degenerate op node over a Const (see
  // TEST_EXTENSION.md): callers don't need to special-case
  // sqrt(p.fixed(2.0)) etc, and validate() never has to reject them.
  friend Expr sqr(Expr a) { return a * a; }

  friend Expr sqrt(Expr a)
  {
    if (a.is_const()) {
      return a.constant(std::sqrt(a.const_value()));
    }
    return {a.graph, a.graph->emit(Op::Sqrt, {a.node_id})};
  }

  friend Expr exp(Expr a)
  {
    if (a.is_const()) {
      return a.constant(std::exp(a.const_value()));
    }
    return {a.graph, a.graph->emit(Op::Exp, {a.node_id})};
  }

  friend Expr log(Expr a)
  {
    if (a.is_const()) {
      return a.constant(std::log(a.const_value()));
    }
    return {a.graph, a.graph->emit(Op::Log, {a.node_id})};
  }

  friend Expr pow(Expr a, int n)
  {
    if (a.is_const()) {
      return a.constant(static_cast<T>(std::pow(a.const_value(), n)));
    }
    DAGNodePayload<T> p;
    p.int_exp = n;
    return {a.graph, a.graph->emit(Op::PowN, {a.node_id}, p)};
  }

  // a^b, general base and exponent -- as opposed to pow(Expr a, int n)'s
  // Op::PowN, which stays a separate op: an integer exponent gets a tighter
  // interval enclosure (repeated squaring) than the general a^b evaluation
  // gives.
  friend Expr pow(Expr a, Expr b)
  {
    if (a.is_const() && b.is_const()) {
      return a.constant(
          static_cast<T>(std::pow(a.const_value(), b.const_value())));
    }
    return {a.graph, a.graph->emit(Op::Pow, {a.node_id, b.node_id})};
  }

  std::size_t id() const { return node_id; }

private:
  ExprDAG<T>* graph;
  std::size_t node_id;

  bool is_const() const { return graph->nodes[node_id].op == Op::Const; }

  T const_value() const { return graph->nodes[node_id].payload.constant; }

  Expr constant(T c) const
  {
    DAGNodePayload<T> p;
    p.constant = c;
    return {graph, graph->emit(Op::Const, {}, p)};
  }
};

/**
 * @brief Reference to the root node of a constraint in the expression DAG
 *
 * @tparam T
 */
template<typename T>
struct ConstraintRef
{
  std::size_t root_id;  // root node
  Cmp cmp;  // constraint type (LE or EQ)
  T rhs;  // RHS of constraint
};

/**
 * @brief Full IR for an MINLP problem
 *
 * Expected usage:
 * @code {.cpp}
 * Problem<double> p;
 * auto x = p.add_variable(-1.0, 1.0);
 * auto y = p.add_variable(0.0, 5.0);
 * auto e = x * x + 2.0 * y;
 * p.set_objective(e);
 * p.add_constraint(x + y, Cmp::LE, 5.0);
 * @endcode
 *
 * @tparam T numerical precision
 */
template<typename T>
struct Problem
{
  Problem() = default;

  // Every Expr handed out by var()/fixed()/etc points at &this->graph. A
  // copy would leave those Exprs aliasing the original's graph while the
  // copy holds an independent one (TEST_EXTENSION.md), so Problem is
  // move-only: copying is a caller mistake worth a compile error rather
  // than silent aliasing.
  Problem(const Problem&) = delete;
  Problem& operator=(const Problem&) = delete;
  Problem(Problem&&) = default;
  Problem& operator=(Problem&&) = default;

  Expr<T> var(T lb, T ub, VarKind kind = VarKind::Continuous)
  {
    std::size_t idx = box_bounds.size();
    box_bounds.push_back(cu::interval<T> {lb, ub});
    var_kinds.push_back(kind);
    DAGNodePayload<T> p;
    p.var_index = idx;
    return Expr<T>(&graph, graph.emit(Op::Var, {}, p));
  }

  Expr<T> var(VarKind kind = VarKind::Continuous)
  {
    return var(
        std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max(), kind);
  }

  Expr<T> int_var(T lb, T ub) { return var(lb, ub, VarKind::Integer); }

  Expr<T> bin_var() { return var(T(0), T(1), VarKind::Binary); }

  /**
   * @brief Emits a constant node holding `value`.
   *
   * Useful for symmetry-broken decision variables: a variable fixed to a
   * known value (to rule out equivalent solutions under a problem symmetry)
   * is represented as a Const node rather than a Var, so it never enters
   * `box_bounds` and is skipped by anything iterating Op::Var nodes.
   */
  Expr<T> fixed(T value)
  {
    DAGNodePayload<T> p;
    p.constant = value;
    return Expr<T>(&graph, graph.emit(Op::Const, {}, p));
  }

  void set_objective(Expr<T> e) { objective_root = e.id(); }

  void add_constraint(Expr<T> lhs, Cmp cmp, T rhs)
  {
    constraints.push_back({lhs.id(), cmp, rhs});
  }

  /**
   * @brief Checks every structural invariant of this Problem and its graph,
   * throwing on the first violation.
   *
   * Deliberately host-only and device-free: every malformed-Problem rejection
   * must be reachable without a CUDA device, so the DAG/Problem test targets
   * don't need a GPU and so callers get a typed error at construction rather
   * than an untyped throw from deep inside GraphBuilder. See
   * TEST_EXTENSION.md for the invariants this covers. Note: an unbounded
   * (or one-sided-bounded) variable is deliberately not rejected here yet --
   * some real problems genuinely don't have both bounds, and the defined
   * partitioning behavior for that case is still an open decision.
   */
  void validate() const
  {
    // graph.nodes bookkeeping -- defense in depth, always true by
    // construction via emit()/var(), but cheap to check and this is the one
    // place callers can rely on it having been checked.
    for (const DAGNode<T>& node : graph.nodes) {
      for (std::size_t in_id : node.in) {
        if (in_id >= node.id) {
          throw InvalidDAG(
              "DAG node operand id is not strictly less than the node's own "
              "id; " "topological-order invariant violated");
        }
      }
      std::size_t expected_op_count = 1;
      for (std::size_t in_id : node.in) {
        expected_op_count += graph.nodes[in_id].op_count;
      }
      if (node.op_count != expected_op_count) {
        throw InvalidDAG(
            "DAG node op_count does not match the size of its subtree");
      }
    }

    std::size_t var_node_count = 0;
    for (const DAGNode<T>& node : graph.nodes) {
      if (node.op == Op::Var) {
        ++var_node_count;
      }
    }
    if (var_node_count != box_bounds.size()
        || box_bounds.size() != var_kinds.size())
    {
      throw InvalidDAG(
          "box_bounds/var_kinds are out of sync with the number of Op::Var "
          "nodes");
    }

    // A Const can never be an objective/constraint root -- unlike a
    // unary op applied to a Const, this isn't foldable into a well-defined
    // value (there's no expression left to evaluate).
    if (objective_root < graph.nodes.size()
        && graph.nodes[objective_root].op == Op::Const)
    {
      throw InvalidDAG(
          "objective root is a bare Const; set_objective was given a fixed() "
          "value " "rather than an expression to optimise");
    }
    for (const ConstraintRef<T>& c : constraints) {
      if (graph.nodes[c.root_id].op == Op::Const) {
        throw InvalidDAG(
            "constraint root is a bare Const; add_constraint was given a "
            "fixed() " "value rather than an expression");
      }
    }

    // Problem-level checks, independent of graph well-formedness.
    if (var_kinds.empty()) {
      throw InvalidProblem("Problem has zero variables");
    }
    if (objective_root == std::numeric_limits<std::size_t>::max()) {
      throw InvalidProblem(
          "Problem's objective was never set (set_objective was never called)");
    }
    for (std::size_t i = 0; i < box_bounds.size(); ++i) {
      const cu::interval<T>& b = box_bounds[i];
      if (!(b.lb <= b.ub)) {
        throw InvalidProblem("variable's lower bound exceeds its upper bound");
      }
      // floor(x) <= x always, so floor(x) < x (rather than !=) is "not
      // already integer-valued" -- also sidesteps -Wfloat-equal.
      if (var_kinds[i] == VarKind::Integer || var_kinds[i] == VarKind::Binary) {
        if (std::floor(b.lb) < b.lb || std::floor(b.ub) < b.ub) {
          throw InvalidProblem(
              "Integer/Binary variable has non-integer-valued bounds");
        }
      }
      if (var_kinds[i] == VarKind::Binary
          && (b.lb < T(0) || b.lb > T(0) || b.ub < T(1) || b.ub > T(1)))
      {
        throw InvalidProblem("Binary variable's bounds are not exactly [0,1]");
      }
    }
    for (const ConstraintRef<T>& c : constraints) {
      if (!std::isfinite(c.rhs)) {
        throw InvalidProblem("constraint rhs must be finite");
      }
    }
  }

  ExprDAG<T> graph;
  std::vector<cu::interval<T>> box_bounds;
  std::vector<VarKind> var_kinds;
  std::size_t objective_root = std::numeric_limits<std::size_t>::max();
  std::vector<ConstraintRef<T>> constraints;
};

}  // namespace cuminlp::model
