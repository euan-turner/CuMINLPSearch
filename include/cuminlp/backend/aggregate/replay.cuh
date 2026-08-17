#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <cuinterval/cuinterval.h>
#include <cuinterval/interval.h>

#include <cub/cub.cuh>

#include "cuminlp/aggregate/bound.hpp"
#include "cuminlp/aggregate/cost.hpp"
#include "cuminlp/backend/aggregate/kernels.cuh"
#include "cuminlp/backend/backend.hpp"
#include "cuminlp/backend/graph/builder.cuh"
#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/backend/graph/kernels.cuh"
#include "cuminlp/cuda_utils.cuh"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/saturating_arith.hpp"

// The aggregate bounder's device side (design/AGGREGATE_BOUNDING.md §3.2,
// §4, §9): one graph over `N = k * M` subregions whose tail reduces them to
// `k` child bounds before anything crosses PCIe.
//
// A separate class from graph::GraphReplay rather than a parameterisation of
// it. That class is built end to end around "one output element per region,
// all of it copied back" -- host staging vectors sized `n_elems`, an ArgMin
// over the whole array, a gathered witness. This one is "N elements in, 2k
// scalars out". Sharing would mean parameterising the tail, the buffers and
// the host staging simultaneously, which is worse than two files that each
// read straight through. Everything below the tail *is* shared: GraphBuilder,
// ops.cuh, and every kernel up to the mask.
namespace cuminlp::backend::aggregate
{

// `cuminlp::aggregate` is a sibling of this namespace, not an enclosing one,
// so unqualified lookup does not reach it -- these are what let the host-side
// role types and the cost model be named plainly below.
using cuminlp::aggregate::AggregateBound;
using cuminlp::aggregate::AggregateBounder;
using cuminlp::aggregate::AggregateRegion;
using cuminlp::aggregate::bounder_element_bytes;
using cuminlp::model::Problem;
using cuminlp::region::Composition;
using cuminlp::region::SlotAssignment;
using cuminlp::region::SlotKind;

/**
 * @brief `N` subregions in, `k` aggregate child bounds out.
 *
 * Built once per (composition, N, k) and reused for every node whose
 * partition has that shape -- `var_ids` and `widths` are per-launch data, as
 * they already are for the existing backend, so any factorisation with
 * `Π widths == N` shares one graph.
 */
template<typename T>
class AggregateBounderReplay : public AggregateBounder<T>
{
public:
  /**
   * @param composition the *launch* composition -- refine slots then branch
   *        slots, in that order (§3.2). Its length fixes this graph's slot
   *        count for life.
   * @param n_regions `N`, the total subregions per launch.
   * @param branch_fan_out `k`. Must divide `N`; `M = N / k` is the segment
   *        length each of the `2k` reductions runs over.
   * @param budget_bytes 0 means "ask the driver what is free".
   */
  static AggregateBounderReplay build(const Problem<T>& problem,
                                      const Composition& composition,
                                      std::size_t n_regions,
                                      std::size_t branch_fan_out,
                                      std::size_t budget_bytes = 0,
                                      bool report_build = false)
  {
    if (branch_fan_out < 1 || n_regions % branch_fan_out != 0) {
      throw cuminlp::InvalidConfiguration(
          "AggregateBounderReplay: the branch fan-out k must divide the total "
          "region count N, so that every child owns exactly M = N / k "
          "contiguous subregions");
    }

    AggregateBounderReplay r;
    r.composition_ = composition;
    r.n_regions_ = n_regions;
    r.k_ = branch_fan_out;
    r.m_ = n_regions / branch_fan_out;
    r.n_vars_ = problem.box_bounds.size();
    r.slot_count_ = composition.size();

    std::size_t const n_buffers = graph::buffer_node_count(problem);
    std::size_t const needed =
        detail::saturating_mul(n_regions, bounder_element_bytes<T>(n_buffers));
    if (budget_bytes == 0) {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      detail::check(cudaMemGetInfo(&free_bytes, &total_bytes),
                    "cudaMemGetInfo");
      budget_bytes = free_bytes;
    }
    if (needed > budget_bytes) {
      throw backend::OverBudgetError(backend::OverBudget {
          .role = "aggregate bounder graph",
          .needed_bytes = needed,
          .budget_bytes = budget_bytes,
          .n_regions = n_regions,
          .elements_per_region = 1,
          .bytes_per_element = bounder_element_bytes<T>(n_buffers),
          .element_breakdown = std::to_string(n_buffers)
              + " DAG-node buffers of "
              + std::to_string(sizeof(cu::interval<T>)) + " B, plus "
              + std::to_string(1 + 4 * sizeof(T))
              + " B of per-element bookkeeping",
          .element_inventory = std::to_string(n_buffers) + " DAG nodes",
          .cost = graph::cost_model_for<T>(n_buffers),
          .composition = composition,
          .fan_out = std::nullopt,
          .solve_samples_per_region = 1,
          .sampler_bytes = 0,
      });
    }

    r.allocate_inputs(problem);
    graph::GraphBuilder<T, cu::interval<T>> builder(problem,
                                                    r.domain_buffer_,
                                                    r.n_regions_,
                                                    r.slot_var_ids_device_,
                                                    r.slot_fan_out_device_,
                                                    r.slot_prefix_device_,
                                                    r.slot_kind_device_,
                                                    r.var_kinds_device_,
                                                    /*sample_points=*/1,
                                                    r.slot_count_);
    r.build_tail(problem, builder);
    r.finish(builder, needed, report_build);
    return r;
  }

  AggregateBounderReplay(const AggregateBounderReplay&) = delete;
  AggregateBounderReplay& operator=(const AggregateBounderReplay&) = delete;

  AggregateBounderReplay(AggregateBounderReplay&& o) noexcept
  {
    *this = std::move(o);
  }

  AggregateBounderReplay& operator=(AggregateBounderReplay&& o) noexcept
  {
    if (this == &o) {
      return *this;
    }
    free_resources();
    graph_ = o.graph_;
    exec_ = o.exec_;
    domain_buffer_ = o.domain_buffer_;
    feasible_buffer_ = o.feasible_buffer_;
    obj_lb_buffer_ = o.obj_lb_buffer_;
    obj_ub_buffer_ = o.obj_ub_buffer_;
    masked_lb_buffer_ = o.masked_lb_buffer_;
    masked_ub_buffer_ = o.masked_ub_buffer_;
    child_lb_buffer_ = o.child_lb_buffer_;
    child_ub_buffer_ = o.child_ub_buffer_;
    cub_temp_ = std::move(o.cub_temp_);
    var_buffers_device_ = o.var_buffers_device_;
    slot_var_ids_device_ = o.slot_var_ids_device_;
    slot_fan_out_device_ = o.slot_fan_out_device_;
    slot_prefix_device_ = o.slot_prefix_device_;
    slot_kind_device_ = o.slot_kind_device_;
    var_kinds_device_ = o.var_kinds_device_;
    node_buffers_ = std::move(o.node_buffers_);
    root_node_ = o.root_node_;
    broadcast_node_ = o.broadcast_node_;
    composition_ = o.composition_;
    n_regions_ = o.n_regions_;
    k_ = o.k_;
    m_ = o.m_;
    n_vars_ = o.n_vars_;
    slot_count_ = o.slot_count_;
    broadcast_grid_ = o.broadcast_grid_;
    apply_grid_ = o.apply_grid_;
    block_ = o.block_;
    allocated_bytes_ = o.allocated_bytes_;
    bounds_host_ = std::move(o.bounds_host_);
    child_lb_host_ = std::move(o.child_lb_host_);
    child_ub_host_ = std::move(o.child_ub_host_);
    retain_subregion_bounds_ = o.retain_subregion_bounds_;
    subregion_lb_host_ = std::move(o.subregion_lb_host_);
    subregion_ub_host_ = std::move(o.subregion_ub_host_);
    subregion_feasible_host_ = std::move(o.subregion_feasible_host_);
    o.graph_ = nullptr;
    o.exec_ = nullptr;
    o.domain_buffer_ = nullptr;
    o.feasible_buffer_ = nullptr;
    o.obj_lb_buffer_ = nullptr;
    o.obj_ub_buffer_ = nullptr;
    o.masked_lb_buffer_ = nullptr;
    o.masked_ub_buffer_ = nullptr;
    o.child_lb_buffer_ = nullptr;
    o.child_ub_buffer_ = nullptr;
    o.var_buffers_device_ = nullptr;
    o.slot_var_ids_device_ = nullptr;
    o.slot_fan_out_device_ = nullptr;
    o.slot_prefix_device_ = nullptr;
    o.slot_kind_device_ = nullptr;
    o.var_kinds_device_ = nullptr;
    return *this;
  }

  ~AggregateBounderReplay() { free_resources(); }

  std::size_t n_subregions() const override { return n_regions_; }

  std::size_t branch_fan_out() const { return k_; }

  std::size_t refine_fan_out() const { return m_; }

  const Composition& composition() const { return composition_; }

  /// What one launch copies back, in bytes. Independent of `N` by
  /// construction -- it is the whole point of the design, and stage 4's
  /// measurable asserts it directly.
  ///
  /// Unless `retain_subregion_bounds` is on, in which case it is deliberately
  /// *not* independent of `N` and says so. See that setter.
  std::size_t d2h_bytes_per_launch() const
  {
    std::size_t bytes = 2 * k_ * sizeof(T);
    if (retain_subregion_bounds_) {
      bytes += n_regions_ * (2 * sizeof(T) + sizeof(unsigned char));
    }
    return bytes;
  }

  /**
   * @brief Also copy the `N` per-subregion bounds back on each launch.
   *
   * Off by default, and must stay off for any solve: the reduction's whole
   * purpose is that only `2k` values cross PCIe regardless of `N`
   * (design/AGGREGATE_BOUNDING.md §2), and copying `N` doubles per launch
   * reintroduces exactly the cost that design exists to remove. The flag
   * exists for design/REFINEMENT_STUDY.md, which measures the distribution
   * of these values and is not a solve.
   *
   * `d2h_bytes_per_launch()` accounts for the extra traffic while this is on,
   * so telemetry does not quietly under-report it.
   */
  void retain_subregion_bounds(bool retain)
  {
    retain_subregion_bounds_ = retain;
    if (retain) {
      subregion_lb_host_.resize(n_regions_);
      subregion_ub_host_.resize(n_regions_);
      subregion_feasible_host_.resize(n_regions_);
    } else {
      subregion_lb_host_.clear();
      subregion_lb_host_.shrink_to_fit();
      subregion_ub_host_.clear();
      subregion_ub_host_.shrink_to_fit();
      subregion_feasible_host_.clear();
      subregion_feasible_host_.shrink_to_fit();
    }
  }

  /// The `N` per-subregion objective bounds the reduction consumed, in
  /// subregion order. Empty unless `retain_subregion_bounds(true)` was called
  /// before the launch. Valid until the next launch.
  ///
  /// These are the *unmasked* values: `subregion_lb()[r]` is region `r`'s raw
  /// interval lower bound whether or not `r` was excluded, so a caller
  /// reproducing `AggregateBound::lb` must filter on `subregion_feasible()`
  /// itself. That is deliberate -- REFINEMENT_STUDY.md §2.5 needs the masked
  /// and unmasked hulls separately, and masking here would discard the
  /// unmasked one irrecoverably.
  std::span<const T> subregion_lb() const { return subregion_lb_host_; }

  std::span<const T> subregion_ub() const { return subregion_ub_host_; }

  /// 1 where region `r`'s relaxation did not prove it infeasible. Note that
  /// "unexcluded" is weaker than "contains a feasible point"
  /// (design/AGGREGATE_BOUNDING.md §1.1).
  std::span<const unsigned char> subregion_feasible() const
  {
    return subregion_feasible_host_;
  }

  /**
   * @brief Device bytes this instance actually requested.
   *
   * Accumulated at the allocation sites, not recomputed from the cost model
   * -- which is the whole point: comparing this against
   * `AggregateBackendFactory::bounder_bytes` checks that §6.3's accounting
   * describes the allocations, and a figure derived from the same formula
   * would check nothing. The DAG-node buffers are counted from the pointers
   * GraphBuilder actually handed over, so `buffer_node_count` predicting that
   * count is part of what is being checked.
   *
   * Measuring this rather than a `cudaMemGetInfo` delta also makes the check
   * deterministic: the test suite runs 16 processes against one device, and
   * free memory moves under all of them.
   */
  std::size_t allocated_bytes() const
  {
    std::size_t buffers = 0;
    for (auto* b : node_buffers_) {
      if (b != nullptr) {
        ++buffers;
      }
    }
    return allocated_bytes_ + buffers * n_regions_ * sizeof(cu::interval<T>);
  }

  std::span<const AggregateBound<T>> bound_children(
      std::span<const AggregateRegion<T>> regions) override
  {
    if (regions.size() != 1) {
      throw cuminlp::InvalidConfiguration(
          "AggregateBounderReplay: batched launches are not built yet "
          "(design/AGGREGATE_BOUNDING.md §14.1); this graph is sized for one "
          "region per launch");
    }
    const AggregateRegion<T>& region = regions[0];
    set_domain(region.box,
               region.partition.launch.var_ids,
               region.partition.launch.widths);
    launch(/*stream=*/0);
    return bounds_host_;
  }

  // Exposed for tests and for the driver's own per-launch validation; the
  // role interface above is what the search actually calls.
  void set_domain(std::span<const cu::interval<T>> domain,
                  std::span<const std::size_t> var_ids,
                  std::span<const std::size_t> widths)
  {
    if (exec_ == nullptr) {
      throw cuminlp::error("set_domain() on a moved-from bounder");
    }
    if (domain.size() != n_vars_) {
      throw cuminlp::ShapeMismatch(
          "set_domain: domain size does not match the problem's variable "
          "count");
    }
    if (var_ids.size() != slot_count_ || widths.size() != slot_count_) {
      throw cuminlp::
          ShapeMismatch(
              "set_domain: var_ids/widths size does not match this graph's "
              "slot " "count");
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
    if (slot_count_ > 0) {
      std::vector<std::uint32_t> fan_out_host(slot_count_);
      for (std::size_t j = 0; j < slot_count_; ++j) {
        fan_out_host[j] = static_cast<std::uint32_t>(widths[j]);
      }
      detail::check(cudaMemcpy(slot_fan_out_device_,
                               fan_out_host.data(),
                               slot_count_ * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy");
      std::vector<std::size_t> const prefix = region::slot_prefixes(widths);
      detail::check(cudaMemcpy(slot_prefix_device_,
                               prefix.data(),
                               slot_count_ * sizeof(std::size_t),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy");
    }

    rebake_root();
  }

  /// Launch, synchronise, and fold the `2k` reduction outputs into `k`
  /// `AggregateBound`s.
  void launch(cudaStream_t stream)
  {
    if (exec_ == nullptr) {
      throw cuminlp::error("launch() on a moved-from bounder");
    }
    detail::check(cudaGraphLaunch(exec_, stream), "cudaGraphLaunch");
    detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    detail::check(cudaMemcpy(child_lb_host_.data(),
                             child_lb_buffer_,
                             k_ * sizeof(T),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    detail::check(cudaMemcpy(child_ub_host_.data(),
                             child_ub_buffer_,
                             k_ * sizeof(T),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");

    // Opt-in, and after the reductions have already consumed these buffers --
    // so what lands on the host is exactly what was reduced, not a separate
    // evaluation that could drift from it.
    if (retain_subregion_bounds_) {
      detail::check(cudaMemcpy(subregion_lb_host_.data(),
                               obj_lb_buffer_,
                               n_regions_ * sizeof(T),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy");
      detail::check(cudaMemcpy(subregion_ub_host_.data(),
                               obj_ub_buffer_,
                               n_regions_ * sizeof(T),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy");
      detail::check(cudaMemcpy(subregion_feasible_host_.data(),
                               feasible_buffer_,
                               n_regions_ * sizeof(unsigned char),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy");
    }

    for (std::size_t c = 0; c < k_; ++c) {
      T const lb = child_lb_host_[c];
      T const ub = child_ub_host_[c];
      // §4.3, resolved without a third reduction: every subregion of this
      // child was excluded exactly when the Max over masked upper bounds is
      // still at its `-inf` identity. `lb == +inf` alone cannot say that --
      // an unbounded relaxation produces it too, and only the first is the
      // `infeasible` claim MINLP_STATUS.md records.
      bool const feasible = ub > -std::numeric_limits<T>::infinity();
      bounds_host_[c] = feasible
          ? AggregateBound<T> {lb, ub, true}
          // Normalised rather than left as (+inf, -inf): an infeasible child
          // is discarded by the driver and never enqueued, and a struct whose
          // hull_ub sits below its lb would trip the frontier's assert if one
          // ever were.
          : AggregateBound<T> {std::numeric_limits<T>::infinity(),
                               std::numeric_limits<T>::infinity(),
                               false};
    }
  }

private:
  AggregateBounderReplay() = default;

  /// alloc_device, plus the running total allocated_bytes() reports.
  template<typename U>
  U* track(std::size_t n)
  {
    allocated_bytes_ += n * sizeof(U);
    return detail::alloc_device<U>(n);
  }

  void allocate_inputs(const Problem<T>& problem)
  {
    domain_buffer_ = track<cu::interval<T>>(n_vars_);
    slot_var_ids_device_ = track<std::size_t>(slot_count_);
    slot_fan_out_device_ = track<std::uint32_t>(slot_count_);
    slot_prefix_device_ = track<std::size_t>(slot_count_);
    slot_kind_device_ = track<SlotKind>(slot_count_);
    var_kinds_device_ = track<model::VarKind>(n_vars_);

    if (slot_count_ > 0) {
      detail::check(cudaMemcpy(slot_kind_device_,
                               composition_.data(),
                               slot_count_ * sizeof(SlotKind),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy");
    }
    detail::check(cudaMemcpy(var_kinds_device_,
                             problem.var_kinds.data(),
                             n_vars_ * sizeof(model::VarKind),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy");
  }

  void build_tail(const Problem<T>& problem,
                  graph::GraphBuilder<T, cu::interval<T>>& builder)
  {
    feasible_buffer_ = track<unsigned char>(n_regions_);
    obj_lb_buffer_ = track<T>(n_regions_);
    obj_ub_buffer_ = track<T>(n_regions_);
    masked_lb_buffer_ = track<T>(n_regions_);
    masked_ub_buffer_ = track<T>(n_regions_);
    child_lb_buffer_ = track<T>(k_);
    child_ub_buffer_ = track<T>(k_);
    child_lb_host_.resize(k_);
    child_ub_host_.resize(k_);
    bounds_host_.resize(k_);

    cudaMemsetParams memset_params {};
    memset_params.dst = feasible_buffer_;
    memset_params.pitch = 0;
    memset_params.value = 1;
    memset_params.elementSize = 1;
    memset_params.width = n_regions_;
    memset_params.height = 1;
    cudaGraphNode_t feasible_memset_node;
    detail::check(
        cudaGraphAddMemsetNode(
            &feasible_memset_node, builder.graph(), nullptr, 0, &memset_params),
        "cudaGraphAddMemsetNode");

    cudaGraphNode_t const obj_producer =
        builder.add_expression(problem.objective_root);

    std::vector<cudaGraphNode_t> deps;
    deps.reserve(problem.constraints.size() + 3);
    for (const auto& constraint : problem.constraints) {
      cudaGraphNode_t const producer =
          builder.add_expression(constraint.root_id);
      deps.push_back(detail::add_kernel_node(
          builder.graph(),
          {producer, feasible_memset_node},
          graph::feasibility_check_kernel<T, cu::interval<T>>,
          builder.grid(),
          builder.block(),
          builder.buffer_for(constraint.root_id),
          constraint.cmp,
          constraint.rhs,
          feasible_buffer_,
          n_regions_));
    }

    deps.push_back(detail::add_kernel_node(
        builder.graph(),
        {obj_producer},
        graph::objective_extract_kernel<T, cu::interval<T>>,
        builder.grid(),
        builder.block(),
        builder.buffer_for(problem.objective_root),
        obj_lb_buffer_,
        n_regions_));
    deps.push_back(detail::add_kernel_node(
        builder.graph(),
        {obj_producer},
        graph::objective_extract_ub_kernel<T, cu::interval<T>>,
        builder.grid(),
        builder.block(),
        builder.buffer_for(problem.objective_root),
        obj_ub_buffer_,
        n_regions_));
    deps.push_back(feasible_memset_node);

    cudaGraphNode_t const mask_node = detail::add_kernel_node(
        builder.graph(),
        deps,
        mask_bounds_kernel<T>,
        builder.grid(),
        builder.block(),
        static_cast<const T*>(obj_lb_buffer_),
        static_cast<const T*>(obj_ub_buffer_),
        static_cast<const unsigned char*>(feasible_buffer_),
        masked_lb_buffer_,
        masked_ub_buffer_,
        n_regions_);

    add_segment_reductions(builder.graph(), mask_node);
  }

  /**
   * @brief The epilogue: `2k` independent reductions, one per child per bound.
   *
   * `k` separate `DeviceReduce` calls rather than one
   * `DeviceSegmentedReduce`: at `k = 4` with `M` in the millions, a segmented
   * reduce sizes its work per segment and would leave the device mostly idle,
   * whereas these are independent child-graph nodes that the graph scheduler
   * overlaps, each saturating on its own `M` elements. `DeviceSegmentedReduce`
   * becomes the better choice if `k` ever grows past a few dozen (§9).
   *
   * CUB dispatches its own internal kernels rather than exposing one nameable
   * `__global__`, so each goes in by stream capture and
   * `cudaGraphAddChildGraphNode` -- the same route the existing backend's
   * ArgMin takes. Build-time cost only.
   */
  void add_segment_reductions(cudaGraph_t graph, cudaGraphNode_t mask_node)
  {
    auto const items = static_cast<std::int64_t>(m_);
    cub_temp_.reserve(2 * k_);

    auto capture = [&](auto&& call)
    {
      cudaStream_t stream;
      detail::check(cudaStreamCreate(&stream), "cudaStreamCreate");
      detail::check(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "cudaStreamBeginCapture");
      call(stream);
      cudaGraph_t captured;
      detail::check(cudaStreamEndCapture(stream, &captured),
                    "cudaStreamEndCapture");
      cudaGraphNode_t node;
      detail::check(
          cudaGraphAddChildGraphNode(&node, graph, &mask_node, 1, captured),
          "cudaGraphAddChildGraphNode");
      detail::check(cudaGraphDestroy(captured), "cudaGraphDestroy");
      detail::check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    };

    for (std::size_t c = 0; c < k_; ++c) {
      const T* const lb_in = masked_lb_buffer_ + c * m_;
      T* const lb_out = child_lb_buffer_ + c;
      std::size_t bytes = 0;
      detail::check(
          cub::DeviceReduce::Min(nullptr, bytes, lb_in, lb_out, items),
          "cub::DeviceReduce::Min (size query)");
      unsigned char* const temp = track<unsigned char>(bytes);
      cub_temp_.push_back(temp);
      capture(
          [&](cudaStream_t s)
          {
            std::size_t n = bytes;
            detail::check(
                cub::DeviceReduce::Min(temp, n, lb_in, lb_out, items, s),
                "cub::DeviceReduce::Min (capture)");
          });

      const T* const ub_in = masked_ub_buffer_ + c * m_;
      T* const ub_out = child_ub_buffer_ + c;
      std::size_t ub_bytes = 0;
      detail::check(
          cub::DeviceReduce::Max(nullptr, ub_bytes, ub_in, ub_out, items),
          "cub::DeviceReduce::Max (size query)");
      unsigned char* const ub_temp = track<unsigned char>(ub_bytes);
      cub_temp_.push_back(ub_temp);
      capture(
          [&](cudaStream_t s)
          {
            std::size_t n = ub_bytes;
            detail::check(
                cub::DeviceReduce::Max(ub_temp, n, ub_in, ub_out, items, s),
                "cub::DeviceReduce::Max (capture)");
          });
    }
  }

  void finish(graph::GraphBuilder<T, cu::interval<T>>& builder,
              std::size_t needed,
              bool report_build)
  {
    graph_ = builder.graph();
    detail::check(cudaGraphInstantiate(&exec_, graph_, 0),
                  "cudaGraphInstantiate");
    root_node_ = builder.root_node();
    broadcast_node_ = builder.broadcast_node();
    var_buffers_device_ = builder.var_buffers_device();
    broadcast_grid_ = builder.broadcast_grid();
    apply_grid_ = builder.apply_grid();
    block_ = builder.block();
    node_buffers_ = builder.take_node_buffers();

    if (report_build) {
      std::cout << "GRAPH\tcomposition=" << region::spell(composition_)
                << "\trole=aggregate-bounder\tregions=" << n_regions_
                << "\tchildren=" << k_ << "\tper_child=" << m_
                << "\tbytes=" << needed << "\td2h=" << d2h_bytes_per_launch()
                << '\n';
    }
  }

  /// The broadcast/apply argument lists have to be re-baked on every
  /// set_domain: the API requires the whole list, not just what changed.
  void rebake_root()
  {
    cudaKernelNodeParams broadcast_params {};
    broadcast_params.gridDim = broadcast_grid_;
    broadcast_params.blockDim = block_;
    broadcast_params.sharedMemBytes = 0;
    broadcast_params.extra = nullptr;
    void* broadcast_args[] = {
        &domain_buffer_, &var_buffers_device_, &n_vars_, &n_regions_};
    broadcast_params.func =
        reinterpret_cast<void*>(graph::broadcast_domain_kernel<T>);
    broadcast_params.kernelParams = broadcast_args;
    detail::check(cudaGraphExecKernelNodeSetParams(
                      exec_, broadcast_node_, &broadcast_params),
                  "cudaGraphExecKernelNodeSetParams");

    if (slot_count_ == 0) {
      return;
    }
    cudaKernelNodeParams apply_params {};
    apply_params.gridDim = apply_grid_;
    apply_params.blockDim = block_;
    apply_params.sharedMemBytes = 0;
    apply_params.extra = nullptr;
    void* apply_args[] = {&domain_buffer_,
                          &slot_var_ids_device_,
                          &slot_fan_out_device_,
                          &slot_prefix_device_,
                          &slot_kind_device_,
                          &var_buffers_device_,
                          &n_regions_,
                          &slot_count_};
    apply_params.func = reinterpret_cast<void*>(graph::apply_slots_kernel<T>);
    apply_params.kernelParams = apply_args;
    detail::check(
        cudaGraphExecKernelNodeSetParams(exec_, root_node_, &apply_params),
        "cudaGraphExecKernelNodeSetParams");
  }

  void free_resources()
  {
    if (exec_) {
      cudaGraphExecDestroy(exec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
    for (void* p : {static_cast<void*>(domain_buffer_),
                    static_cast<void*>(feasible_buffer_),
                    static_cast<void*>(obj_lb_buffer_),
                    static_cast<void*>(obj_ub_buffer_),
                    static_cast<void*>(masked_lb_buffer_),
                    static_cast<void*>(masked_ub_buffer_),
                    static_cast<void*>(child_lb_buffer_),
                    static_cast<void*>(child_ub_buffer_),
                    static_cast<void*>(var_buffers_device_),
                    static_cast<void*>(slot_var_ids_device_),
                    static_cast<void*>(slot_fan_out_device_),
                    static_cast<void*>(slot_prefix_device_),
                    static_cast<void*>(slot_kind_device_),
                    static_cast<void*>(var_kinds_device_)})
    {
      if (p) {
        cudaFree(p);
      }
    }
    for (auto* p : cub_temp_) {
      if (p) {
        cudaFree(p);
      }
    }
    cub_temp_.clear();
    for (auto* p : node_buffers_) {
      if (p) {
        cudaFree(p);
      }
    }
    node_buffers_.clear();
  }

  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  cudaGraphNode_t root_node_ = nullptr;
  cudaGraphNode_t broadcast_node_ = nullptr;
  cu::interval<T>* domain_buffer_ = nullptr;
  unsigned char* feasible_buffer_ = nullptr;
  T* obj_lb_buffer_ = nullptr;
  T* obj_ub_buffer_ = nullptr;
  T* masked_lb_buffer_ = nullptr;
  T* masked_ub_buffer_ = nullptr;
  T* child_lb_buffer_ = nullptr;  ///< k entries
  T* child_ub_buffer_ = nullptr;  ///< k entries
  std::vector<unsigned char*> cub_temp_;  ///< 2k scratch allocations
  cu::interval<T>** var_buffers_device_ = nullptr;
  std::size_t* slot_var_ids_device_ = nullptr;
  std::uint32_t* slot_fan_out_device_ = nullptr;
  std::size_t* slot_prefix_device_ = nullptr;
  SlotKind* slot_kind_device_ = nullptr;
  model::VarKind* var_kinds_device_ = nullptr;
  std::vector<cu::interval<T>*> node_buffers_;

  std::size_t allocated_bytes_ = 0;  ///< this class's own allocations
  std::vector<T> child_lb_host_;
  std::vector<T> child_ub_host_;
  std::vector<AggregateBound<T>> bounds_host_;

  /// design/REFINEMENT_STUDY.md stage 1; empty unless opted in. Host-side
  /// only, so they cost nothing on a solve and are not counted in
  /// allocated_bytes(), which reports *device* bytes.
  bool retain_subregion_bounds_ = false;
  std::vector<T> subregion_lb_host_;
  std::vector<T> subregion_ub_host_;
  std::vector<unsigned char> subregion_feasible_host_;

  Composition composition_ {};
  std::size_t n_regions_ = 0;
  std::size_t k_ = 1;
  std::size_t m_ = 1;
  std::size_t n_vars_ = 0;
  std::size_t slot_count_ = 0;
  dim3 broadcast_grid_ {};
  dim3 apply_grid_ {};
  dim3 block_ {};
};

}  // namespace cuminlp::backend::aggregate
