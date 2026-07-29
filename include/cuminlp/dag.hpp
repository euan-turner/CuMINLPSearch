#pragma once

#include <cuinterval/interval.h>
#include <limits>
#include <vector>

namespace cuminlp::dag {

enum Op { Var, Const, Add, Sub, Mul, Div, Sqr, Neg, Exp, Log, Sqrt, Sin, Cos, Tanh, IPow, Abs, Min, Max };
enum Cmp { LE, EQ };

/**
 * @brief Payload for a DAGNode where the operator requires parameters
 * that are not themselves DAGNodes
 * 
 * @tparam T numerical precision
 */
template<typename T>
union DAGNodePayload {
  T constant;         // payload for Op::Const
  std::size_t var_index; // payload for Op::Var
  int int_exp;         // payload for Op::IPow
};

/**
 * @brief A DAGNode represents a single operation (correspondingly, a variable or intermediate value)
 * in the expression of an objective or constraint.
 * TODO: hash-consing of sub-expressions to collapse
 * 
 * @invariant DAGNode ids are topologically ordered: ∀i∈in:i<id.
 * 
 * @tparam T numerical precision 
 */
template<typename T>
struct DAGNode {
  std::size_t id; // position in expression list, doubles as SSA name
  Op op;       // operator to apply
  std::vector<std::size_t> in; // operator ids (INV: every id in `in` is < id)
  DAGNodePayload<T> payload; // INV: matches Op
  std::size_t op_count; // subtree size of this node + ancestors
  std::size_t live_set_estimate; // peak simultaneously-live interval slots, for later
};

template<typename T>
class ExprDAG {
public:
  std::size_t emit(Op op, std::initializer_list<std::size_t> inputs, DAGNodePayload<T> payload = {}) {
    std::size_t id = nodes.size();
    std::size_t op_count = 1;
    for (auto in_id : inputs) op_count += nodes[in_id].op_count;
    nodes.push_back(DAGNode<T>{id, op, std::vector<std::size_t>(inputs), payload, op_count, 0}); // TODO: liveness checks
    return id;
  }

  std::vector<DAGNode<T>> nodes;
};

/**
 * @brief Expression in the expression DAG, corresponding to variable, constant or operation.
 * 
 * @tparam T 
 */
template<typename T>
class Expr {
public:
  Expr(ExprDAG<T>* g, std::size_t id) : graph(g), node_id(id) {}

  friend Expr operator+(Expr a, Expr b) { return {a.graph, a.graph->emit(Op::Add, {a.node_id, b.node_id})}; }
  friend Expr operator-(Expr a, Expr b) { return {a.graph, a.graph->emit(Op::Sub, {a.node_id, b.node_id})}; }
  friend Expr operator-(Expr a) { return {a.graph, a.graph->emit(Op::Neg, {a.node_id})}; }
  friend Expr operator*(Expr a, Expr b) { 
    if (a.node_id == b.node_id) {
      return {a.graph, a.graph->emit(Op::Sqr, {a.node_id})};
    }
    else {
      return {a.graph, a.graph->emit(Op::Mul, {a.node_id, b.node_id})};
    }
  }
  friend Expr operator/(Expr a, Expr b) { return {a.graph, a.graph->emit(Op::Div, {a.node_id, b.node_id})}; }

  friend Expr operator+(Expr a, T b) { return a + a.constant(b); }
  friend Expr operator+(T a, Expr b) { return b.constant(a) + b; }
  friend Expr operator-(Expr a, T b) { return a - a.constant(b); }
  friend Expr operator-(T a, Expr b) { return b.constant(a) - b; }
  friend Expr operator*(Expr a, T b) { return a * a.constant(b); }
  friend Expr operator*(T a, Expr b) { return b.constant(a) * b; }
  friend Expr operator/(Expr a, T b) { return a / a.constant(b); }
  friend Expr operator/(T a, Expr b) { return b.constant(a) / b; }

  friend Expr sqr(Expr a) { return a * a; }
  friend Expr sqrt(Expr a) { return {a.graph, a.graph->emit(Op::Sqrt, {a.node_id})}; }
  friend Expr exp(Expr a) { return {a.graph, a.graph->emit(Op::Exp, {a.node_id})}; }

  friend Expr pow(Expr a, int n) {
    DAGNodePayload<T> p;
    p.int_exp = n;
    return {a.graph, a.graph->emit(Op::IPow, {a.node_id}, p)};
  }

  std::size_t id() const { return node_id; }

private:
  ExprDAG<T>* graph;
  std::size_t node_id;

  Expr constant(T c) const {
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
struct ConstraintRef {
  std::size_t root_id;   // root node
  Cmp cmp;               // constraint type (LE or EQ)
  T rhs;                 // RHS of constraint
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
struct Problem {
  Expr<T> var(T lb, T ub) {
    std::size_t idx = box_bounds.size();
    box_bounds.push_back(cu::interval<T>{lb, ub});
    DAGNodePayload<T> p;
    p.var_index = idx;
    return Expr<T>(&graph, graph.emit(Op::Var, {}, p));
  }

  Expr<T> var() { return var(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()); }

  Expr<T> lb_var(T lb) { return var(lb, std::numeric_limits<T>::max()); }

  Expr<T> ub_var(T ub) { return var(std::numeric_limits<T>::lowest(), ub); }

  /**
   * @brief Emits a constant node holding `value`.
   *
   * Useful for symmetry-broken decision variables: a variable fixed to a
   * known value (to rule out equivalent solutions under a problem symmetry)
   * is represented as a Const node rather than a Var, so it never enters
   * `box_bounds` and is skipped by anything iterating Op::Var nodes.
   */
  Expr<T> fixed(T value) {
    DAGNodePayload<T> p;
    p.constant = value;
    return Expr<T>(&graph, graph.emit(Op::Const, {}, p));
  }

  void set_objective(Expr<T> e) { objective_root = e.id(); }
  void add_constraint(Expr<T> lhs, Cmp cmp, T rhs) { constraints.push_back({lhs.id(), cmp, rhs}); }

  ExprDAG<T> graph;
  std::vector<cu::interval<T>> box_bounds;
  std::size_t objective_root = std::numeric_limits<std::size_t>::max();
  std::vector<ConstraintRef<T>> constraints;
};

}
