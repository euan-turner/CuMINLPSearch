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
#include "cuminlp/backend/graph/builder.cuh"
#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/backend/graph/kernels.cuh"
#include "cuminlp/backend/graph/ops.cuh"
#include "cuminlp/cuda_utils.cuh"
#include "cuminlp/model/dag.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/decode.hpp"
#include "cuminlp/region/fan_out.hpp"

// GraphReplay: owns the entire graph replay for a problem and value-type V.
// Moved out of graph_replay.cuh (design/MODULE_REFACTOR.md §11).
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

// ---------------------------------------------------------------------------
// Device-memory sizing
//
// These are free functions rather than GraphReplay members because the
// quantity that actually decides whether a configuration runs is not any one
// graph's footprint but the footprint of *every* graph a solve holds at once.
// BackendCache holds, per Composition the search encounters, a point graph, an
// interval graph, and -- when the Composition is fully enumerable -- an exact
// graph, and those are live simultaneously.
//
// Sizing against a single graph is what made the old --max-cycle-size
// suggestion unreachable. The exact graph evaluates one element per region,
// so when it overflowed, its budget scan recommended a cap at which the point
// graph -- sample_points elements per region -- promptly overflowed in turn,
// and the caller got a second out-of-memory failure for having followed the
// advice in the first.
// ---------------------------------------------------------------------------

// buffer_node_count/element_bytes live in backend/graph/cost.hpp (included
// above); auto_max_cycle_size/worst_composition_footprint in
// config/footprint.hpp -- both host-only, so a host-only translation unit
// can reuse them without this file's __global__ kernels and <cub/cub.cuh>.
// See those headers' top comments.

// Which of backend's three roles a (V, Exact) pair plays. The value type
// already decided this -- an interval-valued graph bounds, a point-valued one
// samples, and the deterministic point-valued one enumerates -- so naming the
// role costs one alias and no branching: each instantiation inherits exactly
// the interface it implements, and the compiler rejects any instantiation
// that leaves a pure virtual unimplemented.
template<typename T, typename V, bool Exact>
using replay_role =
    std::conditional_t<!std::is_same_v<V, T>,
                       backend::RegionBounder<T>,
                       std::conditional_t<Exact,
                                          backend::RegionEnumerator<T>,
                                          backend::RegionSampler<T>>>;

// Owns the entire graph replay for a problem and value-type V (either T or
// cu::interval<T>) This includes:
// - The root's broadcast/apply node pair (§4.3)
// - Every op-kernel node
// - A feasibility check kernel per constraint
// - An objective extraction kernel
// Main usage is `build`, `set_domain` and `launch`.
// Each instance owns device memory, a cudaGraph_t and a cudaGraphExec_t
// Exact (only meaningful when V=T) selects the deterministic
// broadcast_domain_point_kernel/apply_slots_point_kernel pair as the root
// instead of the sampling point graph's random-draw pair -- see
// ExactGraphReplay/GraphBuilder.
template<typename T, typename V, bool Exact = false>
class GraphReplay : public replay_role<T, V, Exact>
{
  static constexpr bool is_point = std::is_same_v<V, T>;

public:
  /// @copydoc cuminlp::backend::graph::buffer_node_count
  /// Kept as a member for callers that already have a replay type in hand;
  /// the implementation is shared with the free sizing functions above.
  static std::size_t count_buffer_nodes(const Problem<T>& problem)
  {
    return buffer_node_count(problem);
  }

  /**
   * @brief Device bytes build() would allocate for a composition this wide.
   *
   * Every buffer this class owns is sized off `n_elems` (= n_regions, times
   * sample_points for the point graph), and there is one such buffer per
   * reachable non-Const DAG node. So the total is linear in n_elems with a
   * coefficient the Problem fixes, and can be computed before allocating
   * anything.
   *
   * Saturates at SIZE_MAX, for the same reason composition_fan_out does:
   * with partition_num now a runtime value, `n_regions` can arrive already
   * saturated, and a wrapped byte count would read as comfortably in budget.
   */
  static std::size_t estimate_bytes(const Problem<T>& problem,
                                    std::size_t n_regions,
                                    std::size_t sample_points = 1)
  {
    return bytes_for(n_regions, sample_points, count_buffer_nodes(problem));
  }

  /// @copydoc cuminlp::backend::graph::element_bytes
  static std::size_t bytes_per_element(std::size_t n_buffers)
  {
    return backend::graph::element_bytes<T, V>(n_buffers);
  }

  static std::size_t bytes_for(std::size_t n_regions,
                               std::size_t sample_points,
                               std::size_t n_buffers)
  {
    std::size_t const n_elems =
        is_point ? detail::saturating_mul(n_regions, sample_points) : n_regions;
    return detail::saturating_mul(n_elems, bytes_per_element(n_buffers));
  }

  /// Which of the three graph shapes this instantiation is, for diagnostics.
  static const char* graph_kind()
  {
    if constexpr (is_point && Exact) {
      return "exact";
    } else if constexpr (is_point) {
      return "point";
    } else {
      return "interval";
    }
  }

  /**
   * @brief The facts about a build that would not fit in its budget.
   *
   * Facts only: which role overflowed, how big it was, and what a byte of one
   * element buys here. `config::explain_over_budget` turns these into the
   * human report, including the "try --max-cycle-size=N" half that used to be
   * computed from inside this class by calling the resolver's own scan -- the
   * wrong direction of dependency, and the reason that advice was wrong twice
   * (design/MODULE_REFACTOR.md §5.6, §2.6).
   *
   * `sample_points` is what *this* graph draws and describes the size that
   * failed; `solve_sample_points` is the solve-wide setting, which is what
   * the recommendation has to be costed against. They differ exactly when
   * this is not the point graph.
   */
  static backend::OverBudget over_budget_facts(const Composition& composition,
                                               const FanOutSpec& fan_out,
                                               std::size_t sample_points,
                                               std::size_t n_buffers,
                                               std::size_t needed,
                                               std::size_t budget,
                                               std::size_t solve_sample_points)
  {
    return backend::OverBudget {
        .role = std::string(graph_kind()) + " graph",
        .needed_bytes = needed,
        .budget_bytes = budget,
        .n_regions = composition_fan_out(composition, fan_out),
        .elements_per_region = is_point ? sample_points : 1,
        .bytes_per_element = bytes_per_element(n_buffers),
        .element_breakdown = std::to_string(n_buffers) + " DAG-node buffers of "
            + std::to_string(sizeof(V)) + " B, plus "
            + std::to_string(1 + 3 * sizeof(T))
            + " B of per-element bookkeeping",
        .element_inventory = std::to_string(n_buffers) + " DAG nodes",
        .cost = backend::graph::cost_model_for<T>(n_buffers),
        .composition = composition,
        .fan_out = fan_out,
        .solve_samples_per_region = solve_sample_points,
        .sampler_bytes = 0,
    };
  }

  // `composition` fixes this replay's fan-out/shape for its whole lifetime;
  // only which variables fill its slots (set_domain's `var_ids`) varies
  // between launches.
  //
  // `budget_bytes` caps what this build may allocate on the device; 0 means
  // "ask the driver for what's currently free". A composition whose fan-out
  // exceeds it raises ResourceExhausted *before* allocating anything, rather
  // than dying partway through with a bare cudaErrorMemoryAllocation.
  //
  // `sample_points` is ignored by the interval and exact graphs, which
  // evaluate exactly one element per region; only the sampling point graph
  // reads it.
  static GraphReplay build(const Problem<T>& problem,
                           const Composition& composition,
                           const FanOutSpec& fan_out,
                           std::size_t budget_bytes = 0,
                           std::size_t sample_points = 1)
  {
    if (sample_points < 1) {
      throw cuminlp::InvalidConfiguration(
          "sample_points must be at least 1; a point graph that samples "
          "nothing can never produce an incumbent");
    }

    GraphReplay replay;
    replay.composition_ = composition;
    replay.n_regions_ = composition_fan_out(composition, fan_out);
    replay.n_vars_ = problem.box_bounds.size();
    replay.sample_points_ = is_point && !Exact ? sample_points : 1;
    replay.slot_count_ = composition.size();

    std::size_t const n_buffers = count_buffer_nodes(problem);
    std::size_t const needed =
        bytes_for(replay.n_regions_, replay.sample_points_, n_buffers);
    if (budget_bytes == 0) {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      detail::check(cudaMemGetInfo(&free_bytes, &total_bytes),
                    "cudaMemGetInfo");
      budget_bytes = free_bytes;
    }
    if (needed > budget_bytes) {
      throw backend::OverBudgetError(over_budget_facts(composition,
                                                       fan_out,
                                                       replay.sample_points_,
                                                       n_buffers,
                                                       needed,
                                                       budget_bytes,
                                                       sample_points));
    }

    std::size_t const slot_count = replay.slot_count_;
    replay.domain_buffer_ =
        detail::alloc_device<cu::interval<T>>(replay.n_vars_);
    replay.slot_var_ids_device_ = detail::alloc_device<std::size_t>(slot_count);
    replay.slot_fan_out_device_ =
        detail::alloc_device<std::uint32_t>(slot_count);
    replay.slot_prefix_device_ = detail::alloc_device<std::size_t>(slot_count);
    replay.slot_kind_device_ = detail::alloc_device<SlotKind>(slot_count);
    replay.var_kinds_device_ = detail::alloc_device<VarKind>(replay.n_vars_);

    // fan_out/prefix/kind are fixed by `composition`, and var_kinds by the
    // problem, so they're all uploaded once here rather than on every
    // set_domain() call like var_ids.
    if (slot_count > 0) {
      std::vector<std::uint32_t> fan_out_host(slot_count);
      for (std::size_t j = 0; j < slot_count; ++j) {
        fan_out_host[j] =
            static_cast<std::uint32_t>(slot_fan_out(composition[j], fan_out));
      }
      detail::check(cudaMemcpy(replay.slot_fan_out_device_,
                               fan_out_host.data(),
                               slot_count * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy");
      std::vector<std::size_t> const prefix_host =
          slot_prefixes(composition, fan_out);
      detail::check(cudaMemcpy(replay.slot_prefix_device_,
                               prefix_host.data(),
                               slot_count * sizeof(std::size_t),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy");
      detail::check(cudaMemcpy(replay.slot_kind_device_,
                               composition.data(),
                               slot_count * sizeof(SlotKind),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy");
    }
    detail::check(cudaMemcpy(replay.var_kinds_device_,
                             problem.var_kinds.data(),
                             replay.n_vars_ * sizeof(VarKind),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");

    GraphBuilder<T, V, Exact> builder(problem,
                                      replay.domain_buffer_,
                                      replay.n_regions_,
                                      replay.slot_var_ids_device_,
                                      replay.slot_fan_out_device_,
                                      replay.slot_prefix_device_,
                                      replay.slot_kind_device_,
                                      replay.var_kinds_device_,
                                      replay.sample_points_,
                                      replay.slot_count_);

    // n_elems_ is n_regions_ for the interval graph, n_regions_ * sample_points
    // for the point graph -- every per-node buffer (feasible[], obj_lb[]
    // included) is sized off it, not n_regions_ directly.
    replay.n_elems_ = builder.n_elems();
    replay.feasible_buffer_ =
        detail::alloc_device<unsigned char>(replay.n_elems_);
    replay.obj_lb_buffer_ = detail::alloc_device<T>(replay.n_elems_);
    replay.obj_ub_buffer_ = detail::alloc_device<T>(replay.n_elems_);
    replay.masked_ub_buffer_ = detail::alloc_device<T>(replay.n_elems_);
    replay.candidate_obj_buffer_ = detail::alloc_device<T>(1);
    replay.candidate_index_buffer_ = detail::alloc_device<std::int64_t>(1);
    replay.candidate_point_buffer_ = detail::alloc_device<V>(replay.n_vars_);
    replay.feasible_host_.resize(replay.n_elems_);
    replay.obj_lb_host_.resize(replay.n_elems_);
    replay.candidate_point_host_.resize(replay.n_vars_);

    // Root memset: feasible[] = 1 each replay, race-free
    // fan-out target for every feasibility_check_kernel node.
    cudaMemsetParams memset_params {};
    memset_params.dst = replay.feasible_buffer_;
    memset_params.pitch = 0;
    memset_params.value = 1;
    memset_params.elementSize = 1;
    memset_params.width = replay.n_elems_;
    memset_params.height = 1;
    cudaGraphNode_t feasible_memset_node;
    detail::check(
        cudaGraphAddMemsetNode(
            &feasible_memset_node, builder.graph(), nullptr, 0, &memset_params),
        "cudaGraphAddMemsetNode");

    cudaGraphNode_t obj_producer =
        builder.add_expression(problem.objective_root);

    // Collected so the mask node below can fan in off every constraint's
    // write to feasible[] and ensure serialisation
    std::vector<cudaGraphNode_t> feasibility_nodes;
    feasibility_nodes.reserve(problem.constraints.size());

    for (const auto& constraint : problem.constraints) {
      cudaGraphNode_t constraint_producer =
          builder.add_expression(constraint.root_id);
      cudaGraphNode_t feasibility_node =
          detail::add_kernel_node(builder.graph(),
                                  {constraint_producer, feasible_memset_node},
                                  feasibility_check_kernel<T, V>,
                                  builder.grid(),
                                  builder.block(),
                                  builder.buffer_for(constraint.root_id),
                                  constraint.cmp,
                                  constraint.rhs,
                                  replay.feasible_buffer_,
                                  replay.n_elems_);
      feasibility_nodes.push_back(feasibility_node);
    }

    detail::add_kernel_node(builder.graph(),
                            {obj_producer},
                            objective_extract_kernel<T, V>,
                            builder.grid(),
                            builder.block(),
                            builder.buffer_for(problem.objective_root),
                            replay.obj_lb_buffer_,
                            replay.n_elems_);

    cudaGraphNode_t obj_ub_node =
        detail::add_kernel_node(builder.graph(),
                                {obj_producer},
                                objective_extract_ub_kernel<T, V>,
                                builder.grid(),
                                builder.block(),
                                builder.buffer_for(problem.objective_root),
                                replay.obj_ub_buffer_,
                                replay.n_elems_);

    std::vector<cudaGraphNode_t> mask_deps = feasibility_nodes;
    mask_deps.push_back(feasible_memset_node);
    mask_deps.push_back(obj_ub_node);
    cudaGraphNode_t mask_node =
        detail::add_kernel_node(builder.graph(),
                                mask_deps,
                                mask_infeasible_kernel<T>,
                                builder.grid(),
                                builder.block(),
                                replay.obj_ub_buffer_,
                                replay.feasible_buffer_,
                                replay.masked_ub_buffer_,
                                replay.n_elems_);

    // CUB's DeviceReduce dispatches its own internal kernels rather than
    // exposing one nameable __global__ function, so it can't go through
    // detail::add_kernel_node like everything else here. Instead: capture one
    // call to it on a scratch stream, then fold the captured subgraph in as a
    // single child-graph node. This is a one-time build-time cost, not a
    // per-replay one.
    std::size_t temp_storage_bytes = 0;
    detail::check(
        cub::DeviceReduce::ArgMin(nullptr,
                                  temp_storage_bytes,
                                  replay.masked_ub_buffer_,
                                  replay.candidate_obj_buffer_,
                                  replay.candidate_index_buffer_,
                                  static_cast<std::int64_t>(replay.n_elems_)),
        "cub::DeviceReduce::ArgMin (size query)");
    replay.cub_temp_storage_ =
        detail::alloc_device<unsigned char>(temp_storage_bytes);

    cudaStream_t capture_stream;
    detail::check(cudaStreamCreate(&capture_stream), "cudaStreamCreate");
    detail::check(cudaStreamBeginCapture(capture_stream,
                                         cudaStreamCaptureModeThreadLocal),
                  "cudaStreamBeginCapture");
    detail::check(
        cub::DeviceReduce::ArgMin(replay.cub_temp_storage_,
                                  temp_storage_bytes,
                                  replay.masked_ub_buffer_,
                                  replay.candidate_obj_buffer_,
                                  replay.candidate_index_buffer_,
                                  static_cast<std::int64_t>(replay.n_elems_),
                                  capture_stream),
        "cub::DeviceReduce::ArgMin (capture)");
    cudaGraph_t captured_graph;
    detail::check(cudaStreamEndCapture(capture_stream, &captured_graph),
                  "cudaStreamEndCapture");
    cudaGraphNode_t reduce_node;
    detail::check(
        cudaGraphAddChildGraphNode(
            &reduce_node, builder.graph(), &mask_node, 1, captured_graph),
        "cudaGraphAddChildGraphNode");
    detail::check(cudaGraphDestroy(captured_graph), "cudaGraphDestroy");
    detail::check(cudaStreamDestroy(capture_stream), "cudaStreamDestroy");

    // One thread per variable, reading the index the reduction just wrote, so
    // it must fan in off reduce_node. The var buffers it gathers from are
    // written by root_node_, which reduce_node already transitively depends on.
    // This is the graph's terminal node; nothing downstream depends on it.
    detail::add_kernel_node(
        builder.graph(),
        {reduce_node},
        gather_candidate_point_kernel<T, V>,
        dim3(static_cast<unsigned int>(detail::ceil_div(replay.n_vars_, 256))),
        builder.block(),
        builder.var_buffers_device(),
        static_cast<const std::int64_t*>(replay.candidate_index_buffer_),
        static_cast<const T*>(replay.candidate_obj_buffer_),
        replay.candidate_point_buffer_,
        replay.n_vars_);

    replay.graph_ = builder.graph();
    detail::check(cudaGraphInstantiate(&replay.exec_, replay.graph_, 0),
                  "cudaGraphInstantiate");

    replay.root_node_ = builder.root_node();
    replay.broadcast_node_ = builder.broadcast_node();
    replay.var_buffers_device_ = builder.var_buffers_device();
    replay.broadcast_grid_ = builder.broadcast_grid();
    replay.apply_grid_ = builder.apply_grid();
    replay.block_ = builder.block();
    replay.node_buffers_ = builder.take_node_buffers();

    return replay;
  }

  GraphReplay(const GraphReplay&) = delete;
  GraphReplay& operator=(const GraphReplay&) = delete;

  GraphReplay(GraphReplay&& other) noexcept { *this = std::move(other); }

  GraphReplay& operator=(GraphReplay&& other) noexcept
  {
    if (this == &other) {
      return *this;
    }
    free_resources();
    graph_ = other.graph_;
    exec_ = other.exec_;
    domain_buffer_ = other.domain_buffer_;
    feasible_buffer_ = other.feasible_buffer_;
    obj_lb_buffer_ = other.obj_lb_buffer_;
    obj_ub_buffer_ = other.obj_ub_buffer_;
    masked_ub_buffer_ = other.masked_ub_buffer_;
    candidate_obj_buffer_ = other.candidate_obj_buffer_;
    candidate_index_buffer_ = other.candidate_index_buffer_;
    candidate_point_buffer_ = other.candidate_point_buffer_;
    cub_temp_storage_ = other.cub_temp_storage_;
    var_buffers_device_ = other.var_buffers_device_;
    slot_var_ids_device_ = other.slot_var_ids_device_;
    slot_fan_out_device_ = other.slot_fan_out_device_;
    slot_prefix_device_ = other.slot_prefix_device_;
    slot_kind_device_ = other.slot_kind_device_;
    var_kinds_device_ = other.var_kinds_device_;
    root_node_ = other.root_node_;
    broadcast_node_ = other.broadcast_node_;
    composition_ = other.composition_;
    n_regions_ = other.n_regions_;
    n_elems_ = other.n_elems_;

    sample_points_ = other.sample_points_;
    slot_count_ = other.slot_count_;
    n_vars_ = other.n_vars_;
    broadcast_grid_ = other.broadcast_grid_;
    apply_grid_ = other.apply_grid_;
    block_ = other.block_;
    node_buffers_ = std::move(other.node_buffers_);
    feasible_host_ = std::move(other.feasible_host_);
    obj_lb_host_ = std::move(other.obj_lb_host_);
    candidate_point_host_ = std::move(other.candidate_point_host_);
    candidate_host_ = other.candidate_host_;
    candidate_index_host_ = other.candidate_index_host_;
    other.graph_ = nullptr;
    other.exec_ = nullptr;
    other.domain_buffer_ = nullptr;
    other.feasible_buffer_ = nullptr;
    other.obj_lb_buffer_ = nullptr;
    other.obj_ub_buffer_ = nullptr;
    other.masked_ub_buffer_ = nullptr;
    other.candidate_obj_buffer_ = nullptr;
    other.candidate_index_buffer_ = nullptr;
    other.candidate_point_buffer_ = nullptr;
    other.cub_temp_storage_ = nullptr;
    other.var_buffers_device_ = nullptr;
    other.slot_var_ids_device_ = nullptr;
    other.slot_fan_out_device_ = nullptr;
    other.slot_prefix_device_ = nullptr;
    other.slot_kind_device_ = nullptr;
    other.var_kinds_device_ = nullptr;
    other.root_node_ = nullptr;
    other.broadcast_node_ = nullptr;
    return *this;
  }

  ~GraphReplay() { free_resources(); }

  // Driver calls this before each launch to update the parent domain and
  // which variable fills each slot. Both update via plain memcpy into fixed
  // buffer addresses (slot_fan_out_/slot_prefix_/slot_kind_ don't need
  // re-uploading -- they're fixed by this replay's composition); the kernel
  // args for both the broadcast and apply nodes are re-baked via
  // cudaGraphExecKernelNodeSetParams regardless, since the API requires the
  // whole arg list on every update, not just what changed. `salt` seeds the
  // point graph's sampler (see sample_from_interval); the interval graph
  // takes no such argument and ignores it. `slot_count_ == 0` (a fully
  // resolved box) has no apply node -- root_node_ == broadcast_node_, so
  // only the broadcast rebake below runs.
  void set_domain(std::span<const cu::interval<T>> domain,
                  std::span<const std::size_t> var_ids,
                  std::size_t salt = 0)
  {
    if (exec_ == nullptr) {
      throw cuminlp::error("set_domain() called on a moved-from GraphReplay");
    }
    if (domain.size() != n_vars_) {
      throw cuminlp::ShapeMismatch(
          "set_domain: domain size does not match the problem's variable "
          "count");
    }
    if (var_ids.size() != slot_count_) {
      throw cuminlp::ShapeMismatch(
          "set_domain: var_ids size does not match this replay's slot count");
    }
    detail::check(cudaMemcpy(domain_buffer_,
                             domain.data(),
                             n_vars_ * sizeof(cu::interval<T>),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(slot_var_ids_device_,
                             var_ids.data(),
                             slot_count_ * sizeof(std::size_t),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");

    cudaKernelNodeParams broadcast_params {};
    broadcast_params.gridDim = broadcast_grid_;
    broadcast_params.blockDim = block_;
    broadcast_params.sharedMemBytes = 0;
    broadcast_params.extra = nullptr;

    // The broadcast/apply kernels differ in arity, so the argument arrays do
    // too.
    if constexpr (is_point && Exact) {
      void* broadcast_args[] = {
          &domain_buffer_,
          &var_buffers_device_,
          &n_vars_,
          &n_regions_,
      };
      broadcast_params.func =
          reinterpret_cast<void*>(broadcast_domain_point_kernel<T>);
      broadcast_params.kernelParams = broadcast_args;
      detail::check(cudaGraphExecKernelNodeSetParams(
                        exec_, broadcast_node_, &broadcast_params),
                    "cudaGraphExecKernelNodeSetParams");
    } else if constexpr (is_point) {
      void* broadcast_args[] = {
          &domain_buffer_,
          &var_kinds_device_,
          &var_buffers_device_,
          &n_vars_,
          &n_regions_,
          &sample_points_,
          &salt,
      };
      broadcast_params.func =
          reinterpret_cast<void*>(broadcast_domain_sample_kernel<T>);
      broadcast_params.kernelParams = broadcast_args;
      detail::check(cudaGraphExecKernelNodeSetParams(
                        exec_, broadcast_node_, &broadcast_params),
                    "cudaGraphExecKernelNodeSetParams");
    } else {
      void* broadcast_args[] = {
          &domain_buffer_,
          &var_buffers_device_,
          &n_vars_,
          &n_regions_,
      };
      broadcast_params.func =
          reinterpret_cast<void*>(broadcast_domain_kernel<T>);
      broadcast_params.kernelParams = broadcast_args;
      detail::check(cudaGraphExecKernelNodeSetParams(
                        exec_, broadcast_node_, &broadcast_params),
                    "cudaGraphExecKernelNodeSetParams");
    }

    if (slot_count_ == 0) {
      return;
    }

    cudaKernelNodeParams apply_params {};
    apply_params.gridDim = apply_grid_;
    apply_params.blockDim = block_;
    apply_params.sharedMemBytes = 0;
    apply_params.extra = nullptr;

    if constexpr (is_point && Exact) {
      void* apply_args[] = {
          &domain_buffer_,
          &slot_var_ids_device_,
          &slot_fan_out_device_,
          &slot_prefix_device_,
          &slot_kind_device_,
          &var_buffers_device_,
          &n_regions_,
          &slot_count_,
      };
      apply_params.func = reinterpret_cast<void*>(apply_slots_point_kernel<T>);
      apply_params.kernelParams = apply_args;
      detail::check(
          cudaGraphExecKernelNodeSetParams(exec_, root_node_, &apply_params),
          "cudaGraphExecKernelNodeSetParams");
    } else if constexpr (is_point) {
      void* apply_args[] = {
          &domain_buffer_,
          &slot_var_ids_device_,
          &slot_fan_out_device_,
          &slot_prefix_device_,
          &slot_kind_device_,
          &var_kinds_device_,
          &var_buffers_device_,
          &n_regions_,
          &slot_count_,
          &sample_points_,
          &salt,
      };
      apply_params.func = reinterpret_cast<void*>(apply_slots_sample_kernel<T>);
      apply_params.kernelParams = apply_args;
      detail::check(
          cudaGraphExecKernelNodeSetParams(exec_, root_node_, &apply_params),
          "cudaGraphExecKernelNodeSetParams");
    } else {
      void* apply_args[] = {
          &domain_buffer_,
          &slot_var_ids_device_,
          &slot_fan_out_device_,
          &slot_prefix_device_,
          &slot_kind_device_,
          &var_buffers_device_,
          &n_regions_,
          &slot_count_,
      };
      apply_params.func = reinterpret_cast<void*>(apply_slots_kernel<T>);
      apply_params.kernelParams = apply_args;
      detail::check(
          cudaGraphExecKernelNodeSetParams(exec_, root_node_, &apply_params),
          "cudaGraphExecKernelNodeSetParams");
    }
  }

  // Launches the graph and synchronises; feasible[]/obj_lb[]/candidate()/
  // candidate_point() D2H copy is a manual cudaMemcpy for now. candidate() is
  // this launch's own feasibility-masked min objective upper bound, and
  // candidate_point() the variable values that attained it.
  void launch(cudaStream_t stream)
  {
    if (exec_ == nullptr) {
      throw cuminlp::error("launch() called on a moved-from GraphReplay");
    }
    detail::check(cudaGraphLaunch(exec_, stream), "cudaGraphLaunch");
    detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    detail::check(cudaMemcpy(feasible_host_.data(),
                             feasible_buffer_,
                             n_elems_ * sizeof(unsigned char),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(obj_lb_host_.data(),
                             obj_lb_buffer_,
                             n_elems_ * sizeof(T),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(&candidate_host_,
                             candidate_obj_buffer_,
                             sizeof(T),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(&candidate_index_host_,
                             candidate_index_buffer_,
                             sizeof(std::int64_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(candidate_point_host_.data(),
                             candidate_point_buffer_,
                             n_vars_ * sizeof(V),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
  }

  // ---- backend role interface (design/MODULE_REFACTOR.md §5.1) ----
  //
  // One call each, replacing set_domain + launch + six accessors: the split
  // only ever existed so the driver could read several outputs off one
  // launch, and a result struct of spans says that directly. Exactly one of
  // the three is virtual in any instantiation -- `replay_role` picks which --
  // so the other two are ordinary members that no vtable and no caller of
  // that instantiation ever sees.
  //
  // Every span returned points into this instance's own host-side staging
  // buffers and is valid until the next call on this instance.

  backend::BoundResult<T> bound(const backend::Region<T>& region)
  {
    set_domain(region.box, region.assignment.var_ids);
    launch(/*stream=*/0);
    return backend::BoundResult<T> {feasible_host_, obj_lb_host_, n_regions_};
  }

  backend::CandidateResult<T> sample(const backend::Region<T>& region,
                                     std::uint64_t salt)
  {
    set_domain(region.box, region.assignment.var_ids, salt);
    launch(/*stream=*/0);
    return candidate_result();
  }

  backend::CandidateResult<T> enumerate(const backend::Region<T>& region)
  {
    set_domain(region.box, region.assignment.var_ids);
    launch(/*stream=*/0);
    return candidate_result();
  }

  std::size_t n_samples() const { return n_elems_; }

  /// Points one enumerate() evaluates. Equal to n_regions() by construction:
  /// an exact graph holds one grid point per region.
  std::size_t n_points() const { return n_elems_; }

  std::span<const unsigned char> feasible() const { return feasible_host_; }

  std::span<const T> obj_lb() const { return obj_lb_host_; }

  T candidate() const { return candidate_host_; }

  // The argmin witness for candidate(), indexed by variable: for the point
  // graph the exact sampled values, for the interval graph the argmin sub-box.
  // All-NaN when this launch found no feasible element (candidate() is then
  // CUB's numeric_limits<T>::max() seed) -- check has_candidate() first.
  std::span<const V> candidate_point() const { return candidate_point_host_; }

  // Flat element index the reduction picked; r * sample_points + i for the
  // point graph, the region index for the interval graph.
  std::int64_t candidate_index() const { return candidate_index_host_; }

  bool has_candidate() const
  {
    return candidate_host_ < std::numeric_limits<T>::max();
  }

  std::size_t n_regions() const { return n_regions_; }

  std::size_t n_elems() const { return n_elems_; }

  std::size_t n_vars() const { return n_vars_; }

  const Composition& composition() const { return composition_; }

private:
  GraphReplay() = default;

  // The last launch's reduction, as backend::CandidateResult. `value` is the
  // raw CUB output even when nothing feasible was found (numeric_limits<T>::
  // max()), and `point` is the witness buffer regardless -- all-NaN in that
  // case. Callers check `found` first; the driver's `cand < GUB_` guard is
  // the same test by another name.
  backend::CandidateResult<T> candidate_result() const
  {
    return backend::CandidateResult<T> {
        has_candidate(),
        candidate_host_,
        std::span<const T>(candidate_point_host_),
        candidate_index_host_};
  }

  void free_resources()
  {
    if (exec_) {
      cudaGraphExecDestroy(exec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
    if (domain_buffer_) {
      cudaFree(domain_buffer_);
    }
    if (feasible_buffer_) {
      cudaFree(feasible_buffer_);
    }
    if (obj_lb_buffer_) {
      cudaFree(obj_lb_buffer_);
    }
    if (obj_ub_buffer_) {
      cudaFree(obj_ub_buffer_);
    }
    if (masked_ub_buffer_) {
      cudaFree(masked_ub_buffer_);
    }
    if (candidate_obj_buffer_) {
      cudaFree(candidate_obj_buffer_);
    }
    if (candidate_index_buffer_) {
      cudaFree(candidate_index_buffer_);
    }
    if (candidate_point_buffer_) {
      cudaFree(candidate_point_buffer_);
    }
    if (cub_temp_storage_) {
      cudaFree(cub_temp_storage_);
    }
    if (var_buffers_device_) {
      cudaFree(var_buffers_device_);
    }
    if (slot_var_ids_device_) {
      cudaFree(slot_var_ids_device_);
    }
    if (slot_fan_out_device_) {
      cudaFree(slot_fan_out_device_);
    }
    if (slot_prefix_device_) {
      cudaFree(slot_prefix_device_);
    }
    if (slot_kind_device_) {
      cudaFree(slot_kind_device_);
    }
    if (var_kinds_device_) {
      cudaFree(var_kinds_device_);
    }
    for (auto* buf : node_buffers_) {
      if (buf) {
        cudaFree(buf);
      }
    }
  }

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  cudaGraphNode_t root_node_ = nullptr;  // the apply node, or broadcast_node_
                                         // when slot_count_ == 0
  cudaGraphNode_t broadcast_node_ = nullptr;
  cu::interval<T>* domain_buffer_ = nullptr;  // device, written by set_domain()
  unsigned char* feasible_buffer_ = nullptr;  // device
  T* obj_lb_buffer_ = nullptr;  // device, D2H-copied after each launch()
  T* obj_ub_buffer_ = nullptr;  // device, feeds mask_infeasible_kernel
  T* masked_ub_buffer_ = nullptr;  // device, feeds the CUB reduction
  T* candidate_obj_buffer_ = nullptr;  // device, CUB reduction output (scalar)
  std::int64_t* candidate_index_buffer_ =
      nullptr;  // device, CUB reduction output (flat index)
  V* candidate_point_buffer_ =
      nullptr;  // device, n_vars_, gathered at the argmin index
  unsigned char* cub_temp_storage_ =
      nullptr;  // device, CUB scratch, sized once at build
  V** var_buffers_device_ = nullptr;
  std::size_t* slot_var_ids_device_ =
      nullptr;  // device, slot_count_ entries, written by set_domain()
  std::uint32_t* slot_fan_out_device_ =
      nullptr;  // device, slot_count_ entries, fixed by composition_
  std::size_t* slot_prefix_device_ =
      nullptr;  // device, slot_count_ entries, fixed by composition_
  SlotKind* slot_kind_device_ =
      nullptr;  // device, slot_count_ entries, fixed by composition_
  VarKind* var_kinds_device_ =
      nullptr;  // device, n_vars_ entries, fixed by the problem
  std::vector<V*> node_buffers_;  // every op/Var node's buffer, owned here
  std::vector<unsigned char> feasible_host_;
  std::vector<T> obj_lb_host_;
  std::vector<V> candidate_point_host_;  // D2H-copied after each launch()
  T candidate_host_ =
      std::numeric_limits<T>::infinity();  // D2H-copied after each launch()
  std::int64_t candidate_index_host_ = -1;  // D2H-copied after each launch()
  Composition composition_ {};
  std::size_t n_regions_ = 0;
  std::size_t n_elems_ = 0;

  std::size_t sample_points_ = 1;
  std::size_t slot_count_ = 0;
  std::size_t n_vars_ = 0;
  dim3 broadcast_grid_ {};
  dim3 apply_grid_ {};
  dim3 block_ {};
};

// Convenience aliases. Neither the fan-out widths, the sample count, nor any
// capacity are part of the type any more: they all arrive as build()
// arguments (or, for capacity, don't exist at all), so these differ only in
// the value type V (and hence in which root kernel pair runs).
template<typename T>
using IntervalGraphReplay = GraphReplay<T, cu::interval<T>>;

template<typename T>
using PointGraphReplay = GraphReplay<T, T>;

// Deterministic evaluation over a fully-enumerable Composition: every region
// is an exact grid point rather than a random sample, so its CUB ArgMin
// reduction gives the true minimum objective over every point the
// Composition enumerates -- not a bound, the answer. BackendCache only builds
// and uses one of these for Compositions where is_fully_enumerable() holds,
// and only dispatches to it for a node once every live variable in the box
// has a slot (see SearchDriver::solve()).
template<typename T>
using ExactGraphReplay = GraphReplay<T, T, /*Exact=*/true>;

}  // namespace cuminlp::backend::graph
