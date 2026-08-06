#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuinterval/cuinterval.h>
#include <cuinterval/interval.h>

#include <cub/cub.cuh>
#include <cuda/std/limits>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/backend/graph/kernels.cuh"
#include "cuminlp/backend/graph/ops.cuh"
#include "cuminlp/backend/graph/subregion_sampler.cuh"
#include "cuminlp/cuda_utils.cuh"
#include "cuminlp/model/dag.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/decode.hpp"
#include "cuminlp/region/fan_out.hpp"

// GraphBuilder: wires a Problem's shared ExprDAG into the kernel nodes of
// one CUDA graph. Moved out of graph_replay.cuh (design/MODULE_REFACTOR.md
// §11).
namespace cuminlp::backend::graph
{

// Unqualified names below (SlotKind, Composition, VarKind, Op::..., etc.)
// are the same ones graph_replay.cuh used unqualified before this file
// existed, when they lived in flat cuminlp:: (composition_policy.hpp,
// dag.hpp) and this content sat in namespace cuminlp::dag -- enclosing-scope
// lookup found them without qualification. Now that they live in sibling
// namespaces cuminlp::region / cuminlp::model, that lookup no longer reaches
// them, so these using-declarations restore it explicitly rather than
// re-qualifying every use-site (design/MODULE_REFACTOR.md §11 is a pure
// rename; this keeps the diff a namespace/file move, not a rewrite).
using cuminlp::region::can_fathom_without_children;
using cuminlp::region::Composition;
using cuminlp::region::composition_fan_out;
using cuminlp::region::FanOutSpec;
using cuminlp::region::is_fully_enumerable;
using cuminlp::region::slot_fan_out;
using cuminlp::region::slot_prefixes;
using cuminlp::region::SlotAssignment;
using cuminlp::region::SlotKind;
namespace decode = cuminlp::region::decode;
using cuminlp::model::Cmp;
using cuminlp::model::ConstraintRef;
using cuminlp::model::DAGNode;
using cuminlp::model::ExprDAG;
using cuminlp::model::Op;
using cuminlp::model::Problem;
using cuminlp::model::VarKind;

// Wires a Problem's shared ExprDAG into the kernel nodes of one CUDA graph.
// Contains per-Problem state, shared across each expression (objective or
// constraint), so a node reachable from more than one is allocated and
// evaluated exactly once. V is the value type of a node's buffer, either
// cu::interval<T> for the interval graph, or T for the sample/exact graphs.
// Below the root's broadcast/apply pair, the graphs are identical as the
// kernels and operators are overloaded. Exact (only meaningful when V=T)
// selects the deterministic broadcast/apply_slots_point_kernel pair instead
// of the sampling point graph's broadcast/apply_slots_sample_kernel pair --
// see ExactGraphReplay.
template<typename T, typename V, bool Exact = false>
class GraphBuilder
{
  static_assert(
      std::is_same_v<V, T> || std::is_same_v<V, cu::interval<T>>,
      "V must be T (point/exact graph) or cu::interval<T> (interval graph)");
  static constexpr bool is_point = std::is_same_v<V, T>;

public:
  /**
   * @brief Construct a new Graph Builder object.
   *
   * @param problem Problem representation
   * @param domain_buffer Storage for domain being evaluated
   * @param n_regions Number of regions (the composition's fan-out)
   * @param slot_var_ids Device buffer, slot_count entries: which variable
   * each slot acts on
   * @param slot_fan_out Device buffer, slot_count entries: fan-out of each
   * slot
   * @param slot_prefix Device buffer, slot_count entries: prefix product of
   * each slot's fan-out (see slot_prefixes, region/composition.hpp)
   * @param slot_kind Device buffer, slot_count entries: operation each slot
   * performs
   * @param var_kinds Device buffer, n_vars entries: per-variable kind. Only
   *                  consumed by the point graph's root nodes (so samples
   *                  for Integer/Binary variables land on the integer
   *                  lattice); the interval graph ignores it.
   * @param sample_points Points sampled per subdomain. Meaningful only for
   *                  the sampling point graph; the interval and exact graphs
   *                  evaluate one element per region and pass 1.
   */
  GraphBuilder(const Problem<T>& problem,
               cu::interval<T>* domain_buffer,
               std::size_t n_regions,
               const std::size_t* slot_var_ids,
               const std::uint32_t* slot_fan_out,
               const std::size_t* slot_prefix,
               const SlotKind* slot_kind,
               const VarKind* var_kinds,
               std::size_t sample_points,
               std::size_t slot_count)
      : problem_(problem)
      , n_regions_(n_regions)
      , sample_points_(sample_points)
      , slot_count_(slot_count)
      , n_elems_(
            is_point
                ? n_regions * sample_points
                : n_regions)  // the number of V-typed slots to operate over
      , block_(256)
      , grid_(static_cast<unsigned int>(detail::ceil_div(n_elems_, 256)))
      , broadcast_grid_(static_cast<unsigned int>(
            detail::ceil_div(problem_.box_bounds.size() * n_elems_, 256)))
      , apply_grid_(static_cast<unsigned int>(
            detail::ceil_div(slot_count_ * n_regions_, 256)))
  {
    detail::check(cudaGraphCreate(&graph_, 0), "cudaGraphCreate");

    buffers_.resize(problem_.graph.nodes.size(), nullptr);
    producer_nodes_.resize(problem_.graph.nodes.size(), nullptr);

    // Every Op::Var node is materialised eagerly by the root's broadcast/
    // apply pair, which scatters the parent domain (narrowed by any live
    // slot) into each variable's buffer. Op::Const nodes get no buffer;
    // their payload is consumed by value at the use site (see wire_binary).
    std::size_t n_vars = problem_.box_bounds.size();
    std::vector<V*> var_buffer_list(n_vars, nullptr);
    for (const auto& node : problem_.graph.nodes) {
      if (node.op == Op::Var) {
        V* buf = detail::alloc_device<V>(n_elems_);
        var_buffer_list[node.payload.var_index] = buf;
        buffers_[node.id] = buf;
      }
    }

    var_buffers_device_ = detail::alloc_device<V*>(n_vars);
    detail::check(cudaMemcpy(var_buffers_device_,
                             var_buffer_list.data(),
                             n_vars * sizeof(V*),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");

    // Broadcast, then apply the live slots on top -- an explicit dependency
    // edge so the narrowing cannot race the broadcast (§4.3). root_node_ is
    // the apply node: it is what every Op::Var producer, and hence every
    // downstream node, actually depends on. slot_count_ == 0 (a fully
    // resolved box) has nothing to apply, so the broadcast alone is root.
    cudaGraphNode_t broadcast_node;
    if constexpr (is_point && Exact) {
      broadcast_node = detail::add_kernel_node(graph_,
                                               {},
                                               broadcast_domain_point_kernel<T>,
                                               broadcast_grid_,
                                               block_,
                                               domain_buffer,
                                               var_buffers_device_,
                                               n_vars,
                                               n_regions_);
      root_node_ = slot_count_ == 0
          ? broadcast_node
          : detail::add_kernel_node(graph_,
                                    {broadcast_node},
                                    apply_slots_point_kernel<T>,
                                    apply_grid_,
                                    block_,
                                    domain_buffer,
                                    slot_var_ids,
                                    slot_fan_out,
                                    slot_prefix,
                                    slot_kind,
                                    var_buffers_device_,
                                    n_regions_,
                                    slot_count_);
    } else if constexpr (is_point) {
      broadcast_node =
          detail::add_kernel_node(graph_,
                                  {},
                                  broadcast_domain_sample_kernel<T>,
                                  broadcast_grid_,
                                  block_,
                                  domain_buffer,
                                  var_kinds,
                                  var_buffers_device_,
                                  n_vars,
                                  n_regions_,
                                  sample_points_,
                                  std::size_t {0});
      root_node_ = slot_count_ == 0
          ? broadcast_node
          : detail::add_kernel_node(graph_,
                                    {broadcast_node},
                                    apply_slots_sample_kernel<T>,
                                    apply_grid_,
                                    block_,
                                    domain_buffer,
                                    slot_var_ids,
                                    slot_fan_out,
                                    slot_prefix,
                                    slot_kind,
                                    var_kinds,
                                    var_buffers_device_,
                                    n_regions_,
                                    slot_count_,
                                    sample_points_,
                                    std::size_t {0});
    } else {
      broadcast_node = detail::add_kernel_node(graph_,
                                               {},
                                               broadcast_domain_kernel<T>,
                                               broadcast_grid_,
                                               block_,
                                               domain_buffer,
                                               var_buffers_device_,
                                               n_vars,
                                               n_regions_);
      root_node_ = slot_count_ == 0
          ? broadcast_node
          : detail::add_kernel_node(graph_,
                                    {broadcast_node},
                                    apply_slots_kernel<T>,
                                    apply_grid_,
                                    block_,
                                    domain_buffer,
                                    slot_var_ids,
                                    slot_fan_out,
                                    slot_prefix,
                                    slot_kind,
                                    var_buffers_device_,
                                    n_regions_,
                                    slot_count_);
    }
    broadcast_node_ = broadcast_node;

    for (const auto& node : problem_.graph.nodes) {
      if (node.op == Op::Var) {
        producer_nodes_[node.id] = root_node_;
      }
    }
  }

  // Idempotent: ensures node `id`'s buffer and producing kernel node exist,
  // recursing into `.in` first. Returns the producing graph node (the shared
  // partition node for Op::Var, or the op kernel node otherwise).
  cudaGraphNode_t ensure_node(std::size_t id)
  {
    if (producer_nodes_[id] != nullptr) {
      return producer_nodes_[id];
    }

    const DAGNode<T>& node = problem_.graph.nodes[id];
    switch (node.op) {
      case Op::Const:
        throw std::runtime_error("ensure_node called on an Op::Const node; constants are consumed "
                                 "by value at their use site, never given their own graph node");
      case Op::Var:
        throw std::runtime_error("Op::Var node missing its producer; GraphBuilder constructor "
                                 "invariant broken");
      case Op::Add:
        producer_nodes_[id] = wire_binary<AddOp>(id);
        break;
      case Op::Sub:
        producer_nodes_[id] = wire_binary<SubOp>(id);
        break;
      case Op::Mul:
        producer_nodes_[id] = wire_binary<MulOp>(id);
        break;
      case Op::Div:
        producer_nodes_[id] = wire_binary<DivOp>(id);
        break;
      case Op::Min:
        producer_nodes_[id] = wire_binary<MinOp>(id);
        break;
      case Op::Max:
        producer_nodes_[id] = wire_binary<MaxOp>(id);
        break;
      case Op::Pow:
        producer_nodes_[id] = wire_binary<PowOp>(id);
        break;
      case Op::Neg:
        producer_nodes_[id] = wire_unary<NegOp>(id);
        break;
      case Op::Sqr:
        producer_nodes_[id] = wire_unary<SqrOp>(id);
        break;
      case Op::Exp:
        producer_nodes_[id] = wire_unary<ExpOp>(id);
        break;
      case Op::Log:
        producer_nodes_[id] = wire_unary<LogOp>(id);
        break;
      case Op::Sqrt:
        producer_nodes_[id] = wire_unary<SqrtOp>(id);
        break;
      case Op::Sin:
        producer_nodes_[id] = wire_unary<SinOp>(id);
        break;
      case Op::Cos:
        producer_nodes_[id] = wire_unary<CosOp>(id);
        break;
      case Op::Tanh:
        producer_nodes_[id] = wire_unary<TanhOp>(id);
        break;
      case Op::Abs:
        producer_nodes_[id] = wire_unary<AbsOp>(id);
        break;
      case Op::PowN:
        producer_nodes_[id] = wire_pown(id);
        break;
    }
    return producer_nodes_[id];
  }

  // Walks every node reachable from `root_id` (one function's expression)
  // and wires it into the graph, returning the node producing root_id's
  // buffer.
  cudaGraphNode_t add_expression(std::size_t root_id)
  {
    if (problem_.graph.nodes[root_id].op == Op::Const) {
      throw std::runtime_error("add_expression called on an Op::Const root; not producible via "
                               "the current Expr API (constant() is private, only ever emitted as "
                               "an immediate operand of a binary op)");
    }
    return ensure_node(root_id);
  }

  V* buffer_for(std::size_t id) const { return buffers_[id]; }

  cudaGraph_t graph() const { return graph_; }

  cudaGraphNode_t root_node() const { return root_node_; }

  // The broadcast node, distinct from root_node() (the apply node) whenever
  // this Composition has at least one live slot. Equal to root_node() when
  // slot_count_ == 0 (a fully resolved box), since there is then nothing to
  // apply.
  cudaGraphNode_t broadcast_node() const { return broadcast_node_; }

  V** var_buffers_device() const { return var_buffers_device_; }

  dim3 grid() const { return grid_; }

  dim3 broadcast_grid() const { return broadcast_grid_; }

  dim3 apply_grid() const { return apply_grid_; }

  dim3 block() const { return block_; }

  std::size_t n_elems() const { return n_elems_; }

  // Yields ownership of the per-node buffers to the caller (GraphReplay).
  // GraphBuilder itself never frees anything.
  std::vector<V*> take_node_buffers() { return std::move(buffers_); }

private:
  template<class BinaryOp>
  cudaGraphNode_t wire_binary(std::size_t id)
  {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    if (node.op != op_code<BinaryOp>::value) {
      throw std::runtime_error("wire_binary<BinaryOp> called on a node whose op does not match "
                               "BinaryOp; ensure_node's dispatch switch is out of sync with the "
                               "op_code<> trait");
    }
    if (node.in.size() != 2) {
      throw std::runtime_error(
          "wire_binary called on a node without exactly two operands");
    }
    std::size_t lhs_id = node.in[0];
    std::size_t rhs_id = node.in[1];
    bool lhs_const = problem_.graph.nodes[lhs_id].op == Op::Const;
    bool rhs_const = problem_.graph.nodes[rhs_id].op == Op::Const;

    buffers_[id] = detail::alloc_device<V>(n_elems_);

    if (!lhs_const && !rhs_const) {
      cudaGraphNode_t lhs_node = ensure_node(lhs_id);
      cudaGraphNode_t rhs_node = ensure_node(rhs_id);
      // Both operands can share the same producer (e.g. `x + y` where x and y
      // are both Op::Var, produced by the one shared partition node), and
      // cudaGraphAddKernelNode rejects a duplicated entry in the dependency
      // list, so dedup before adding.
      std::vector<cudaGraphNode_t> deps = (lhs_node == rhs_node)
          ? std::vector<cudaGraphNode_t> {lhs_node}
          : std::vector<cudaGraphNode_t> {lhs_node, rhs_node};
      return detail::add_kernel_node(graph_,
                                     deps,
                                     binary_op_kernel<V, T, BinaryOp>,
                                     grid_,
                                     block_,
                                     buffers_[lhs_id],
                                     buffers_[rhs_id],
                                     buffers_[id],
                                     n_elems_);
    }
    if (!lhs_const && rhs_const) {
      cudaGraphNode_t lhs_node = ensure_node(lhs_id);
      T rhs_val = problem_.graph.nodes[rhs_id].payload.constant;
      return detail::add_kernel_node(
          graph_,
          {lhs_node},
          binary_op_scalar_rhs_kernel<V, T, BinaryOp>,
          grid_,
          block_,
          buffers_[lhs_id],
          rhs_val,
          buffers_[id],
          n_elems_);
    }
    if (lhs_const && !rhs_const) {
      T lhs_val = problem_.graph.nodes[lhs_id].payload.constant;
      cudaGraphNode_t rhs_node = ensure_node(rhs_id);
      return detail::add_kernel_node(
          graph_,
          {rhs_node},
          binary_op_scalar_lhs_kernel<V, T, BinaryOp>,
          grid_,
          block_,
          lhs_val,
          buffers_[rhs_id],
          buffers_[id],
          n_elems_);
    }
    // Both operands are Op::Const (e.g. two Problem::fixed() values combined
    // directly): the result doesn't depend on the domain or any other node,
    // so it's materialised with no dependencies, same as root_node_.
    T lhs_val = problem_.graph.nodes[lhs_id].payload.constant;
    T rhs_val = problem_.graph.nodes[rhs_id].payload.constant;
    return detail::add_kernel_node(
        graph_,
        {},
        binary_op_scalar_scalar_kernel<V, T, BinaryOp>,
        grid_,
        block_,
        lhs_val,
        rhs_val,
        buffers_[id],
        n_elems_);
  }

  template<class UnaryOp>
  cudaGraphNode_t wire_unary(std::size_t id)
  {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    if (node.op != op_code<UnaryOp>::value) {
      throw std::runtime_error("wire_unary<UnaryOp> called on a node whose op does not match "
                               "UnaryOp; ensure_node's dispatch switch is out of sync with the "
                               "op_code<> trait");
    }
    if (node.in.size() != 1) {
      throw std::runtime_error(
          "wire_unary called on a node without exactly one operand");
    }
    std::size_t operand_id = node.in[0];
    if (problem_.graph.nodes[operand_id].op == Op::Const) {
      throw std::runtime_error("unary op applied directly to an Op::Const; unreachable via the "
                               "current Expr API");
    }
    cudaGraphNode_t operand_node = ensure_node(operand_id);
    buffers_[id] = detail::alloc_device<V>(n_elems_);
    return detail::add_kernel_node(graph_,
                                   {operand_node},
                                   unary_op_kernel<V, T, UnaryOp>,
                                   grid_,
                                   block_,
                                   buffers_[operand_id],
                                   buffers_[id],
                                   n_elems_);
  }

  cudaGraphNode_t wire_pown(std::size_t id)
  {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    if (node.op != Op::PowN) {
      throw std::runtime_error(
          "wire_pown called on a node whose op is not Op::PowN");
    }
    if (node.in.size() != 1) {
      throw std::runtime_error(
          "wire_pown called on a node without exactly one operand");
    }
    std::size_t operand_id = node.in[0];
    if (problem_.graph.nodes[operand_id].op == Op::Const) {
      throw std::runtime_error("Op::PowN applied directly to an Op::Const; unreachable via the "
                               "current Expr API");
    }
    cudaGraphNode_t operand_node = ensure_node(operand_id);
    buffers_[id] = detail::alloc_device<V>(n_elems_);
    return detail::add_kernel_node(graph_,
                                   {operand_node},
                                   pown_kernel<T, V>,
                                   grid_,
                                   block_,
                                   buffers_[operand_id],
                                   node.payload.int_exp,
                                   buffers_[id],
                                   n_elems_);
  }

  const Problem<T>& problem_;
  std::size_t n_regions_;
  // A member, not a ctor local: add_kernel_node takes the address of what it
  // is given, so the value must outlive the call that bakes it into the
  // graph node.
  std::size_t sample_points_;
  std::size_t slot_count_;
  std::size_t n_elems_;
  dim3 block_;
  dim3 grid_;
  dim3 broadcast_grid_;
  dim3 apply_grid_;
  cudaGraph_t graph_ {};
  cudaGraphNode_t broadcast_node_ {};
  cudaGraphNode_t root_node_ {};
  V** var_buffers_device_ = nullptr;
  std::vector<V*> buffers_;  // indexed by node id, null until allocated
  std::vector<cudaGraphNode_t>
      producer_nodes_;  // indexed by node id, null until added
};

}  // namespace cuminlp::backend::graph
