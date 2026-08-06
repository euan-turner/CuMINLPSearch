#pragma once

#include <cstddef>
#include <vector>

// The expression IR: an SSA-numbered DAG of scalar operations. `model/
// problem.hpp` builds `Expr`/`Problem` on top of this; nothing here knows
// what a "variable" or "constraint" is.
namespace cuminlp::model
{

enum class Op
{
  Var,
  Const,
  Add,
  Sub,
  Mul,
  Div,
  Sqr,
  Neg,
  Exp,
  Log,
  Sqrt,
  Sin,
  Cos,
  Tanh,
  PowN,
  Pow,
  Abs,
  Min,
  Max
};

/**
 * @brief Payload for a DAGNode where the operator requires parameters
 * that are not themselves DAGNodes
 *
 * @tparam T numerical precision
 */
template<typename T>
union DAGNodePayload
{
  T constant;  // payload for Op::Const
  std::size_t var_index;  // payload for Op::Var
  int int_exp;  // payload for Op::PowN
};

/**
 * @brief A DAGNode represents a single operation (correspondingly, a variable
 * or intermediate value) in the expression of an objective or constraint.
 * TODO: hash-consing of sub-expressions to collapse
 *
 * @invariant DAGNode ids are topologically ordered: ∀i∈in:i<id.
 *
 * @tparam T numerical precision
 */
template<typename T>
struct DAGNode
{
  std::size_t id;  // position in expression list, doubles as SSA name
  Op op;  // operator to apply
  std::vector<std::size_t> in;  // operator ids (INV: every id in `in` is < id)
  DAGNodePayload<T> payload;  // INV: matches Op
  std::size_t op_count;  // subtree size of this node + ancestors
  std::size_t
      live_set_estimate;  // peak simultaneously-live interval slots, for later
};

template<typename T>
class ExprDAG
{
public:
  std::size_t emit(Op op,
                   std::initializer_list<std::size_t> inputs,
                   DAGNodePayload<T> payload = {})
  {
    std::size_t id = nodes.size();
    std::size_t op_count = 1;
    for (auto in_id : inputs) {
      op_count += nodes[in_id].op_count;
    }
    nodes.push_back(DAGNode<T> {id,
                                op,
                                std::vector<std::size_t>(inputs),
                                payload,
                                op_count,
                                0});  // TODO: liveness checks
    return id;
  }

  std::vector<DAGNode<T>> nodes;
};

}  // namespace cuminlp::model
