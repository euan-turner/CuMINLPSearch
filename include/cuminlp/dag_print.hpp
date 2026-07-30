#pragma once

/**
 * @file
 * @brief Human-readable rendering of a Problem's expression DAG.
 *
 * `Problem::validate()` proves a DAG is *well-formed*; nothing proved it was
 * the *right* DAG. The only semantic oracle in the repo is the CPU evaluator
 * in test/source/dag_eval.hpp, which compares a parsed Problem against a
 * hand-built one -- an oracle that exists for exactly two MINLPLib instances.
 * For everything else, reading the expression back is the only check there is,
 * so this header exists to make that possible without a debugger.
 *
 * Deliberately core rather than part of the GAMS frontend: it renders
 * dag::Problem, which knows nothing about GAMS. Variable names are an optional
 * caller-supplied side table (ParsedModel::var_names is indexed to match), so
 * a hand-built Problem can be dumped just as well as a parsed one.
 *
 * Host-only and header-only: no CUDA, no device code, matching dag.hpp's own
 * "a malformed Problem must be diagnosable without a GPU" stance.
 */

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "cuminlp/dag.hpp"

namespace cuminlp::dag
{

/// How much of the DAG to render.
enum class PrintStyle
{
  /// Reconstructed infix expression, one line per root. Readable, and directly
  /// comparable against the source model; loses the sharing that makes this a
  /// DAG rather than a tree, and can be exponentially large if there is a lot
  /// of it. Use `Nodes` on anything big.
  Infix,
  /// One line per node in topological (SSA) order. Linear in the DAG's actual
  /// size and shows shared subexpressions exactly once, which is the whole
  /// point of the representation.
  Nodes
};

struct PrintOptions
{
  PrintStyle style = PrintStyle::Infix;

  /**
   * Give up on an Infix root once its subtree exceeds this many operators,
   * printing a placeholder instead. `op_count` makes the check free, and
   * without it a single dump of a large instance can bury a terminal in
   * megabytes of parentheses. Zero disables the limit.
   */
  std::size_t max_infix_ops = 4000;

  /// Names for variables, indexed like `Problem::box_bounds`. Shorter than the
  /// variable count (or empty) is fine: unnamed variables print as `x<i>`.
  std::vector<std::string> var_names;
};

inline auto op_name(Op op) -> char const*
{
  switch (op) {
    case Op::Var:
      return "var";
    case Op::Const:
      return "const";
    case Op::Add:
      return "add";
    case Op::Sub:
      return "sub";
    case Op::Mul:
      return "mul";
    case Op::Div:
      return "div";
    case Op::Sqr:
      return "sqr";
    case Op::Neg:
      return "neg";
    case Op::Exp:
      return "exp";
    case Op::Log:
      return "log";
    case Op::Sqrt:
      return "sqrt";
    case Op::Sin:
      return "sin";
    case Op::Cos:
      return "cos";
    case Op::Tanh:
      return "tanh";
    case Op::PowN:
      return "pown";
    case Op::Pow:
      return "pow";
    case Op::Abs:
      return "abs";
    case Op::Min:
      return "min";
    case Op::Max:
      return "max";
  }
  return "?";
}

inline auto var_kind_name(VarKind kind) -> char const*
{
  switch (kind) {
    case VarKind::Continuous:
      return "continuous";
    case VarKind::Integer:
      return "integer";
    case VarKind::Binary:
      return "binary";
  }
  return "?";
}

inline auto cmp_name(Cmp cmp) -> char const*
{
  switch (cmp) {
    case Cmp::LE:
      return "<=";
    case Cmp::EQ:
      return "==";
  }
  return "?";
}

// Not `detail`: graph_replay.cuh is also inside cuminlp::dag and calls
// `detail::check` unqualified, expecting to find cuminlp::detail. A
// cuminlp::dag::detail would hide it and break every CUDA translation unit
// that includes both.
namespace print_detail
{

inline auto variable_name(PrintOptions const& options,
                          std::size_t index) -> std::string
{
  if (index < options.var_names.size() && !options.var_names[index].empty()) {
    return options.var_names[index];
  }
  return "x" + std::to_string(index);
}

/// Infix spelling of a binary operator, or nullptr if it has none and must be
/// printed in function-call form.
inline auto infix_symbol(Op op) -> char const*
{
  switch (op) {
    case Op::Add:
      return " + ";
    case Op::Sub:
      return " - ";
    case Op::Mul:
      return " * ";
    case Op::Div:
      return " / ";
    default:
      return nullptr;
  }
}

template<typename T>
void write_infix(std::ostream& os,
                 ExprDAG<T> const& graph,
                 std::size_t id,
                 PrintOptions const& options)
{
  DAGNode<T> const& node = graph.nodes[id];

  switch (node.op) {
    case Op::Var:
      os << variable_name(options, node.payload.var_index);
      return;
    case Op::Const:
      os << node.payload.constant;
      return;
    case Op::PowN:
      // The exponent lives in the payload, not in `in`, so this can't fall
      // through to the generic n-ary case below.
      os << '(';
      write_infix(os, graph, node.in[0], options);
      os << ')' << '^' << node.payload.int_exp;
      return;
    default:
      break;
  }

  if (char const* symbol = infix_symbol(node.op); symbol && node.in.size() == 2)
  {
    os << '(';
    write_infix(os, graph, node.in[0], options);
    os << symbol;
    write_infix(os, graph, node.in[1], options);
    os << ')';
    return;
  }

  os << op_name(node.op) << '(';
  for (std::size_t i = 0; i < node.in.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    write_infix(os, graph, node.in[i], options);
  }
  os << ')';
}

template<typename T>
void write_root(std::ostream& os,
                ExprDAG<T> const& graph,
                std::size_t id,
                PrintOptions const& options)
{
  if (id >= graph.nodes.size()) {
    os << "<unset>";
    return;
  }
  if (options.max_infix_ops > 0
      && graph.nodes[id].op_count > options.max_infix_ops)
  {
    os << "<" << graph.nodes[id].op_count << " operators; too large to render "
       << "as infix, use the nodes style>";
    return;
  }
  write_infix(os, graph, id, options);
}

/// One `%id = op operands  [ops=n]` line per node, in topological order.
template<typename T>
void write_nodes(std::ostream& os,
                 ExprDAG<T> const& graph,
                 PrintOptions const& options)
{
  os << "\nnodes (" << graph.nodes.size() << "):\n";
  for (DAGNode<T> const& node : graph.nodes) {
    os << "  %" << node.id << " = " << op_name(node.op);
    switch (node.op) {
      case Op::Var:
        os << ' ' << variable_name(options, node.payload.var_index);
        break;
      case Op::Const:
        os << ' ' << node.payload.constant;
        break;
      case Op::PowN:
        // Exponent is payload, not an operand, so it can't print like one.
        os << " %" << node.in[0] << ", " << node.payload.int_exp;
        break;
      default:
        for (std::size_t i = 0; i < node.in.size(); ++i) {
          os << (i == 0 ? " " : ", ") << '%' << node.in[i];
        }
        break;
    }
    os << "  [ops=" << node.op_count << "]\n";
  }
}

/// Renders a root: a `%id` reference in the Nodes style, the reconstructed
/// expression in the Infix style.
template<typename T>
void write_root_styled(std::ostream& os,
                       ExprDAG<T> const& graph,
                       std::size_t id,
                       PrintOptions const& options)
{
  if (options.style != PrintStyle::Nodes) {
    write_root(os, graph, id, options);
    return;
  }
  if (id >= graph.nodes.size()) {
    os << "<unset>";
  } else {
    os << '%' << id;
  }
}

}  // namespace print_detail

/**
 * @brief Writes `problem` to `os` in the style `options` selects.
 *
 * Renders the box, the objective and every constraint. Nothing here mutates
 * or validates the Problem -- a malformed one still prints, which is the
 * point when the reason it is malformed is what you are looking for. The one
 * exception is a root id past the end of the node list (an objective that was
 * never set), which prints as `<unset>` rather than reading out of bounds.
 */
template<typename T>
void print_problem(std::ostream& os,
                   Problem<T> const& problem,
                   PrintOptions const& options = {})
{
  os << "variables (" << problem.box_bounds.size() << "):\n";
  for (std::size_t i = 0; i < problem.box_bounds.size(); ++i) {
    char const* kind = i < problem.var_kinds.size()
        ? var_kind_name(problem.var_kinds[i])
        : "?";
    os << "  " << print_detail::variable_name(options, i) << " : " << kind
       << " in [" << problem.box_bounds[i].lb << ", "
       << problem.box_bounds[i].ub << "]\n";
  }

  if (options.style == PrintStyle::Nodes) {
    print_detail::write_nodes(os, problem.graph, options);
  }

  os << "\nobjective:\n  ";
  print_detail::write_root_styled(
      os, problem.graph, problem.objective_root, options);
  os << '\n';

  os << "\nconstraints (" << problem.constraints.size() << "):\n";
  for (std::size_t i = 0; i < problem.constraints.size(); ++i) {
    ConstraintRef<T> const& c = problem.constraints[i];
    os << "  [" << i << "] ";
    print_detail::write_root_styled(os, problem.graph, c.root_id, options);
    os << ' ' << cmp_name(c.cmp) << ' ' << c.rhs << '\n';
  }
}

}  // namespace cuminlp::dag
