#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuinterval/interval.h>
#include <cuinterval/cuinterval.h>
#include "cuda_utils.cuh"
#include "dag.hpp"
#include "partition.cuh"

namespace cuminlp::dag {

  /**
   * @brief Partitions the `parent_domain` into materialised sub-domains
   *
   * @tparam T 
   * @tparam CycleSize 
   * @tparam PartitionNum 
   * @param parent_domain 
   * @param cycle_start 
   * @param var_buffers 
   * @param n_vars 
   * @param n_regions 
   * @return __global__ 
   */
template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
__global__ void partition_variables_kernel(
  const cu::interval<T>* __restrict__ parent_domain, // NUM_VARS (box bounds per variable)
  std::size_t cycle_start,
  cu::interval<T>* const* __restrict__ var_buffers, // NUM_VARS x NUM_REGIONS (box bounds per variable per region)
  std::size_t n_vars,
  std::size_t n_regions
) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t r = tid; r < n_regions; r += num_threads) {
    auto ctx = partition::make_cycle_context<CycleSize, PartitionNum>(r, cycle_start);
    for (std::size_t vid = 0; vid < n_vars; ++vid) {
      cu::interval<T> v;
      partition::get_bounds(ctx, parent_domain, vid, v);
      var_buffers[vid][r] = v;
    }
  }
}

// Functor wrappers for Op

// Each tag overloads apply() for interval-interval, interval-scalar, and
// scalar-interval operands. min/max lack a mixed overload in cuinterval, so
// their scalar apply() wraps the scalar as a point interval.
struct AddOp {
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b) { return a + b; }
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, T b) { return a + b; }
  template<typename T> static __device__ cu::interval<T> apply(T a, cu::interval<T> b) { return a + b; }
};
struct SubOp {
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b) { return a - b; }
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, T b) { return a - b; }
  template<typename T> static __device__ cu::interval<T> apply(T a, cu::interval<T> b) { return a - b; }
};
struct MulOp {
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b) { return a * b; }
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, T b) { return a * b; }
  template<typename T> static __device__ cu::interval<T> apply(T a, cu::interval<T> b) { return a * b; }
};
struct DivOp {
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b) { return a / b; }
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, T b) { return a / b; }
  template<typename T> static __device__ cu::interval<T> apply(T a, cu::interval<T> b) { return a / b; }
};
struct MinOp {
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b) { return min(a, b); }
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, T b) { return min(a, cu::interval<T>(b)); }
  template<typename T> static __device__ cu::interval<T> apply(T a, cu::interval<T> b) { return min(cu::interval<T>(a), b); }
};
struct MaxOp {
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b) { return max(a, b); }
  template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a, T b) { return max(a, cu::interval<T>(b)); }
  template<typename T> static __device__ cu::interval<T> apply(T a, cu::interval<T> b) { return max(cu::interval<T>(a), b); }
};

struct NegOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return -a; } };
struct SqrOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return sqr(a); } };
struct ExpOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return exp(a); } };
struct LogOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return log(a); } };
struct SqrtOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return sqrt(a); } };
struct SinOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return sin(a); } };
struct CosOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return cos(a); } };
struct TanhOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return tanh(a); } };
struct AbsOp { template<typename T> static __device__ cu::interval<T> apply(cu::interval<T> a) { return abs(a); } };


// Each of the kernels below applies the templated operation across its input intervals, and stores the result in the output
template<typename T, class UnaryOp>
__global__ void unary_op_kernel(const cu::interval<T>* __restrict__ a,
                                cu::interval<T>* __restrict out,
                                std::size_t n_regions) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = UnaryOp::apply(a[i]);
  }
}

template<typename T, class BinaryOp>
__global__ void binary_op_kernel(const cu::interval<T>* __restrict__ a,
                                const cu::interval<T>* __restrict__ b,
                                cu::interval<T>* __restrict out,
                                std::size_t n_regions) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = BinaryOp::apply(a[i], b[i]);
  }
}

// b is an Op::Const operand's payload, passed by value -- never materialised
// into an n_regions buffer
template<typename T, class BinaryOp>
__global__ void binary_op_scalar_rhs_kernel(const cu::interval<T>* __restrict__ a,
                                            T b,
                                            cu::interval<T>* __restrict__ out,
                                            std::size_t n_regions) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = BinaryOp::apply(a[i], b);
  }
}

// a is an Op::Const operand's payload, passed by value -- never materialised
// into an n_regions buffer
template<typename T, class BinaryOp>
__global__ void binary_op_scalar_lhs_kernel(T a,
                                            const cu::interval<T>* __restrict__ b,
                                            cu::interval<T>* __restrict__ out,
                                            std::size_t n_regions) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = BinaryOp::apply(a, b[i]);
  }
}

template<typename T>
__global__ void ipow_kernel(const cu::interval<T>* __restrict__ a,
                            int exponent,
                            cu::interval<T>* __restrict__ out,
                            std::size_t n_regions) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    out[i] = pown(a[i], exponent);
  }
}

// Writes the shared, absorbing-zero feasible[] flag; concurrent writes agree,
// so no atomics needed. A region is feasible if any part of its LHS bound
// can satisfy the constraint.
template<typename T>
__global__ void feasibility_check_kernel(const cu::interval<T>* __restrict__ lhs,
                                         Cmp cmp,
                                         T rhs,
                                         unsigned char* __restrict__ feasible,
                                         std::size_t n_regions) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    cu::interval<T> v = lhs[i];
    bool ok = false;
    if (cmp == Cmp::LE && v.lb <= rhs) {
      ok = true;
    } else if (cmp == Cmp::EQ && v.lb <= rhs && rhs <= v.ub) {
      ok = true;
    }
    if (!ok) {
      feasible[i] = 0;
    }
  }
}

// Extracts obj's lower bound into obj_lb[] for every region, feasible or
// not -- interval soundness holds regardless of constraint feasibility.
template<typename T>
__global__ void objective_extract_kernel(const cu::interval<T>* __restrict__ obj,
                                         T* __restrict__ obj_lb,
                                         std::size_t n_regions) {
  std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t num_threads = gridDim.x * blockDim.x;

  for (std::size_t i = tid; i < n_regions; i += num_threads) {
    obj_lb[i] = obj[i].lb;
  }
}

// Wires a Problem's shared ExprDAG into the kernel nodes of one CUDA graph.
// Per-Problem construction state, shared across every expression (objective
// and constraints) so a node reachable from more than one is allocated and
// evaluated exactly once. `detail::` below resolves to cuminlp::detail
// (cuda_utils.cuh).
template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
class GraphBuilder {
public:
  GraphBuilder(const Problem<T>& problem, cu::interval<T>* domain_buffer, std::size_t n_regions)
      : problem_(problem)
      , n_regions_(n_regions)
      , block_(256)
      , grid_(static_cast<unsigned int>(detail::ceil_div(n_regions, 256)))
  {
    detail::check(cudaGraphCreate(&graph_, 0), "cudaGraphCreate");

    buffers_.resize(problem_.graph.nodes.size(), nullptr);
    producer_nodes_.resize(problem_.graph.nodes.size(), nullptr);

    // Every Op::Var node is materialised eagerly in one shared kernel node
    // that partitions the parent domain and scatters it into each variable's
    // buffer. Op::Const nodes get no buffer; their payload is consumed by
    // value at the use site (see wire_binary).
    std::size_t n_vars = problem_.box_bounds.size();
    std::vector<cu::interval<T>*> var_buffer_list(n_vars, nullptr);
    for (const auto& node : problem_.graph.nodes) {
      if (node.op == Op::Var) {
        cu::interval<T>* buf = detail::alloc_device<cu::interval<T>>(n_regions_);
        var_buffer_list[node.payload.var_index] = buf;
        buffers_[node.id] = buf;
      }
    }

    var_buffers_device_ = detail::alloc_device<cu::interval<T>*>(n_vars);
    detail::check(cudaMemcpy(var_buffers_device_, var_buffer_list.data(),
                             n_vars * sizeof(cu::interval<T>*), cudaMemcpyHostToDevice),
                 "cudaMemcpy");

    partition_node_ = detail::add_kernel_node(
        graph_, {}, partition_variables_kernel<T, CycleSize, PartitionNum>, grid_, block_,
        domain_buffer, std::size_t {0}, var_buffers_device_, n_vars, n_regions_);

    for (const auto& node : problem_.graph.nodes) {
      if (node.op == Op::Var) producer_nodes_[node.id] = partition_node_;
    }
  }

  // Idempotent: ensures node `id`'s buffer and producing kernel node exist,
  // recursing into `.in` first. Returns the producing graph node (the shared
  // partition node for Op::Var, or the op kernel node otherwise). Never
  // called on Op::Const (see wire_binary).
  cudaGraphNode_t ensure_node(std::size_t id) {
    if (producer_nodes_[id] != nullptr) return producer_nodes_[id];

    const DAGNode<T>& node = problem_.graph.nodes[id];
    switch (node.op) {
      case Op::Const:
        throw std::runtime_error("ensure_node called on an Op::Const node; constants are consumed "
                                 "by value at their use site, never given their own graph node");
      case Op::Var:
        throw std::runtime_error("Op::Var node missing its producer; GraphBuilder constructor "
                                 "invariant broken");
      case Op::Add: producer_nodes_[id] = wire_binary<AddOp>(id); break;
      case Op::Sub: producer_nodes_[id] = wire_binary<SubOp>(id); break;
      case Op::Mul: producer_nodes_[id] = wire_binary<MulOp>(id); break;
      case Op::Div: producer_nodes_[id] = wire_binary<DivOp>(id); break;
      case Op::Min: producer_nodes_[id] = wire_binary<MinOp>(id); break;
      case Op::Max: producer_nodes_[id] = wire_binary<MaxOp>(id); break;
      case Op::Neg:  producer_nodes_[id] = wire_unary<NegOp>(id); break;
      case Op::Sqr:  producer_nodes_[id] = wire_unary<SqrOp>(id); break;
      case Op::Exp:  producer_nodes_[id] = wire_unary<ExpOp>(id); break;
      case Op::Log:  producer_nodes_[id] = wire_unary<LogOp>(id); break;
      case Op::Sqrt: producer_nodes_[id] = wire_unary<SqrtOp>(id); break;
      case Op::Sin:  producer_nodes_[id] = wire_unary<SinOp>(id); break;
      case Op::Cos:  producer_nodes_[id] = wire_unary<CosOp>(id); break;
      case Op::Tanh: producer_nodes_[id] = wire_unary<TanhOp>(id); break;
      case Op::Abs:  producer_nodes_[id] = wire_unary<AbsOp>(id); break;
      case Op::IPow: producer_nodes_[id] = wire_ipow(id); break;
    }
    return producer_nodes_[id];
  }

  // Walks every node reachable from `root_id` (one function's expression)
  // and wires it into the graph, returning the node producing root_id's
  // buffer.
  cudaGraphNode_t add_expression(std::size_t root_id) {
    if (problem_.graph.nodes[root_id].op == Op::Const) {
      throw std::runtime_error("add_expression called on an Op::Const root; not producible via "
                               "the current Expr API (constant() is private, only ever emitted as "
                               "an immediate operand of a binary op)");
    }
    return ensure_node(root_id);
  }

  cu::interval<T>* buffer_for(std::size_t id) const { return buffers_[id]; }
  cudaGraph_t graph() const { return graph_; }
  cudaGraphNode_t partition_node() const { return partition_node_; }
  cu::interval<T>** var_buffers_device() const { return var_buffers_device_; }
  dim3 grid() const { return grid_; }
  dim3 block() const { return block_; }

  // Yields ownership of the per-node buffers to the caller (GraphReplay).
  // GraphBuilder itself never frees anything.
  std::vector<cu::interval<T>*> take_node_buffers() { return std::move(buffers_); }

private:
  template<class BinaryOp>
  cudaGraphNode_t wire_binary(std::size_t id) {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    std::size_t lhs_id = node.in[0];
    std::size_t rhs_id = node.in[1];
    bool lhs_const = problem_.graph.nodes[lhs_id].op == Op::Const;
    bool rhs_const = problem_.graph.nodes[rhs_id].op == Op::Const;

    buffers_[id] = detail::alloc_device<cu::interval<T>>(n_regions_);

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
      return detail::add_kernel_node(graph_, deps, binary_op_kernel<T, BinaryOp>,
                                     grid_, block_, buffers_[lhs_id], buffers_[rhs_id],
                                     buffers_[id], n_regions_);
    }
    if (!lhs_const && rhs_const) {
      cudaGraphNode_t lhs_node = ensure_node(lhs_id);
      T rhs_val = problem_.graph.nodes[rhs_id].payload.constant;
      return detail::add_kernel_node(graph_, {lhs_node}, binary_op_scalar_rhs_kernel<T, BinaryOp>,
                                     grid_, block_, buffers_[lhs_id], rhs_val, buffers_[id],
                                     n_regions_);
    }
    if (lhs_const && !rhs_const) {
      T lhs_val = problem_.graph.nodes[lhs_id].payload.constant;
      cudaGraphNode_t rhs_node = ensure_node(rhs_id);
      return detail::add_kernel_node(graph_, {rhs_node}, binary_op_scalar_lhs_kernel<T, BinaryOp>,
                                     grid_, block_, lhs_val, buffers_[rhs_id], buffers_[id],
                                     n_regions_);
    }
    throw std::runtime_error("both operands of a binary op are Op::Const; unreachable via the "
                             "current Expr API");
  }

  template<class UnaryOp>
  cudaGraphNode_t wire_unary(std::size_t id) {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    std::size_t operand_id = node.in[0];
    if (problem_.graph.nodes[operand_id].op == Op::Const) {
      throw std::runtime_error("unary op applied directly to an Op::Const; unreachable via the "
                               "current Expr API");
    }
    cudaGraphNode_t operand_node = ensure_node(operand_id);
    buffers_[id] = detail::alloc_device<cu::interval<T>>(n_regions_);
    return detail::add_kernel_node(graph_, {operand_node}, unary_op_kernel<T, UnaryOp>, grid_,
                                   block_, buffers_[operand_id], buffers_[id], n_regions_);
  }

  cudaGraphNode_t wire_ipow(std::size_t id) {
    const DAGNode<T>& node = problem_.graph.nodes[id];
    std::size_t operand_id = node.in[0];
    if (problem_.graph.nodes[operand_id].op == Op::Const) {
      throw std::runtime_error("Op::IPow applied directly to an Op::Const; unreachable via the "
                               "current Expr API");
    }
    cudaGraphNode_t operand_node = ensure_node(operand_id);
    buffers_[id] = detail::alloc_device<cu::interval<T>>(n_regions_);
    return detail::add_kernel_node(graph_, {operand_node}, ipow_kernel<T>, grid_, block_,
                                   buffers_[operand_id], node.payload.int_exp, buffers_[id],
                                   n_regions_);
  }

  const Problem<T>& problem_;
  std::size_t n_regions_;
  dim3 block_;
  dim3 grid_;
  cudaGraph_t graph_ {};
  cudaGraphNode_t partition_node_ {};
  cu::interval<T>** var_buffers_device_ = nullptr;
  std::vector<cu::interval<T>*> buffers_;        // indexed by node id, null until allocated
  std::vector<cudaGraphNode_t> producer_nodes_;  // indexed by node id, null until added
};

// Owns one Problem's whole graph replay: the partition root, every op-kernel
// node, one feasibility_check_kernel per constraint, the objective_extract_kernel
// node, and the buffers the driver reads/writes each replay. No device-side
// GUB reduction yet -- GUB comes from the driver's sample-based loop instead.
// Move-only: owns device memory, a cudaGraph_t and a cudaGraphExec_t.
template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
class GraphReplay {
public:
  static GraphReplay build(const Problem<T>& problem, std::size_t max_regions) {
    GraphReplay replay;
    replay.n_regions_ = max_regions;
    replay.n_vars_ = problem.box_bounds.size();

    replay.domain_buffer_ = detail::alloc_device<cu::interval<T>>(replay.n_vars_);
    replay.feasible_buffer_ = detail::alloc_device<unsigned char>(replay.n_regions_);
    replay.obj_lb_buffer_ = detail::alloc_device<T>(replay.n_regions_);
    replay.feasible_host_.resize(replay.n_regions_);
    replay.obj_lb_host_.resize(replay.n_regions_);

    GraphBuilder<T, CycleSize, PartitionNum> builder(problem, replay.domain_buffer_,
                                                     replay.n_regions_);

    // Root memset: feasible[] = 1 each replay, race-free
    // fan-out target for every feasibility_check_kernel node.
    cudaMemsetParams memset_params {};
    memset_params.dst = replay.feasible_buffer_;
    memset_params.pitch = 0;
    memset_params.value = 1;
    memset_params.elementSize = 1;
    memset_params.width = replay.n_regions_;
    memset_params.height = 1;
    cudaGraphNode_t feasible_memset_node;
    detail::check(cudaGraphAddMemsetNode(&feasible_memset_node, builder.graph(), nullptr, 0,
                                        &memset_params),
                 "cudaGraphAddMemsetNode");

    cudaGraphNode_t obj_producer = builder.add_expression(problem.objective_root);

    for (const auto& constraint : problem.constraints) {
      cudaGraphNode_t constraint_producer = builder.add_expression(constraint.root_id);
      detail::add_kernel_node(
          builder.graph(), {constraint_producer, feasible_memset_node}, feasibility_check_kernel<T>,
          builder.grid(), builder.block(), builder.buffer_for(constraint.root_id), constraint.cmp,
          constraint.rhs, replay.feasible_buffer_, replay.n_regions_);
    }

    detail::add_kernel_node(builder.graph(), {obj_producer}, objective_extract_kernel<T>,
                            builder.grid(), builder.block(),
                            builder.buffer_for(problem.objective_root), replay.obj_lb_buffer_,
                            replay.n_regions_);

    replay.graph_ = builder.graph();
    detail::check(cudaGraphInstantiate(&replay.exec_, replay.graph_, 0), "cudaGraphInstantiate");

    replay.partition_node_ = builder.partition_node();
    replay.var_buffers_device_ = builder.var_buffers_device();
    replay.grid_ = builder.grid();
    replay.block_ = builder.block();
    replay.node_buffers_ = builder.take_node_buffers();

    return replay;
  }

  GraphReplay(const GraphReplay&) = delete;
  GraphReplay& operator=(const GraphReplay&) = delete;

  GraphReplay(GraphReplay&& other) noexcept { *this = std::move(other); }

  GraphReplay& operator=(GraphReplay&& other) noexcept {
    if (this == &other) return *this;
    free_resources();
    graph_ = other.graph_;
    exec_ = other.exec_;
    domain_buffer_ = other.domain_buffer_;
    feasible_buffer_ = other.feasible_buffer_;
    obj_lb_buffer_ = other.obj_lb_buffer_;
    var_buffers_device_ = other.var_buffers_device_;
    partition_node_ = other.partition_node_;
    n_regions_ = other.n_regions_;
    n_vars_ = other.n_vars_;
    grid_ = other.grid_;
    block_ = other.block_;
    node_buffers_ = std::move(other.node_buffers_);
    feasible_host_ = std::move(other.feasible_host_);
    obj_lb_host_ = std::move(other.obj_lb_host_);
    other.graph_ = nullptr;
    other.exec_ = nullptr;
    other.domain_buffer_ = nullptr;
    other.feasible_buffer_ = nullptr;
    other.obj_lb_buffer_ = nullptr;
    other.var_buffers_device_ = nullptr;
    other.partition_node_ = nullptr;
    return *this;
  }

  ~GraphReplay() { free_resources(); }

  // Driver calls this before each launch to update the parent domain and
  // dimension-cycling offset (Section 11's "set domain" entry point). domain
  // updates via plain memcpy (fixed buffer address, topology unchanged);
  // cycle_start is a baked-in kernel arg, updated via
  // cudaGraphExecKernelNodeSetParams (Section 9.5).
  void set_domain(std::span<const cu::interval<T>> domain, std::size_t cycle_start) {
    if (domain.size() != n_vars_) {
      throw std::runtime_error("set_domain: domain size does not match the problem's variable count");
    }
    detail::check(cudaMemcpy(domain_buffer_, domain.data(), n_vars_ * sizeof(cu::interval<T>),
                             cudaMemcpyHostToDevice),
                 "cudaMemcpy");

    void* kernel_args[] = {
        const_cast<void*>(static_cast<const void*>(&domain_buffer_)),
        const_cast<void*>(static_cast<const void*>(&cycle_start)),
        const_cast<void*>(static_cast<const void*>(&var_buffers_device_)),
        const_cast<void*>(static_cast<const void*>(&n_vars_)),
        const_cast<void*>(static_cast<const void*>(&n_regions_)),
    };
    cudaKernelNodeParams params {};
    params.func = reinterpret_cast<void*>(partition_variables_kernel<T, CycleSize, PartitionNum>);
    params.gridDim = grid_;
    params.blockDim = block_;
    params.sharedMemBytes = 0;
    params.kernelParams = kernel_args;
    params.extra = nullptr;

    detail::check(cudaGraphExecKernelNodeSetParams(exec_, partition_node_, &params),
                 "cudaGraphExecKernelNodeSetParams");
  }

  // Launches the graph and synchronises; feasible[]/obj_lb[] D2H copy is a
  // manual cudaMemcpy for now (Section 9.1 step 5 moves this in-graph later).
  void launch(cudaStream_t stream) {
    detail::check(cudaGraphLaunch(exec_, stream), "cudaGraphLaunch");
    detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    detail::check(cudaMemcpy(feasible_host_.data(), feasible_buffer_,
                             n_regions_ * sizeof(unsigned char), cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
    detail::check(cudaMemcpy(obj_lb_host_.data(), obj_lb_buffer_, n_regions_ * sizeof(T),
                             cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
  }

  std::span<const unsigned char> feasible() const { return feasible_host_; }
  std::span<const T> obj_lb() const { return obj_lb_host_; }
  std::size_t n_regions() const { return n_regions_; }

private:
  GraphReplay() = default;

  void free_resources() {
    if (exec_) cudaGraphExecDestroy(exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (domain_buffer_) cudaFree(domain_buffer_);
    if (feasible_buffer_) cudaFree(feasible_buffer_);
    if (obj_lb_buffer_) cudaFree(obj_lb_buffer_);
    if (var_buffers_device_) cudaFree(var_buffers_device_);
    for (auto* buf : node_buffers_) {
      if (buf) cudaFree(buf);
    }
  }

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  cudaGraphNode_t partition_node_ = nullptr;
  cu::interval<T>* domain_buffer_ = nullptr;   // device, written by set_domain()
  unsigned char* feasible_buffer_ = nullptr;   // device
  T* obj_lb_buffer_ = nullptr;                 // device, D2H-copied after each launch()
  cu::interval<T>** var_buffers_device_ = nullptr;
  std::vector<cu::interval<T>*> node_buffers_; // every op/Var node's buffer, owned here
  std::vector<unsigned char> feasible_host_;
  std::vector<T> obj_lb_host_;
  std::size_t n_regions_ = 0;
  std::size_t n_vars_ = 0;
  dim3 grid_ {};
  dim3 block_ {};
};

}