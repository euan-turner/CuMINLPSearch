#pragma once

#include <cstddef>

#include <cuinterval/interval.h>

#include "cuminlp/backend/cost_model.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/saturating_arith.hpp"
#include "cuminlp/search_sizing.hpp"

// What the CUDA-graph backend costs, expressed as a backend::RegionCostModel
// (design/MODULE_REFACTOR.md §5.5).
//
// Host-only on purpose, and the reason it is a separate header from
// graph_replay.cuh: the resolver's shape fit runs against these coefficients
// in a test target with no GPU and no CUDA toolchain.
namespace cuminlp::backend::graph
{

/// Bytes one element of a V-valued graph costs: one V per buffer-bearing
/// node, plus feasible[] (1 byte) and obj_lb/obj_ub/masked_ub (T each). The
/// interval graph is the expensive one, at sizeof(cu::interval<T>) per node
/// against the point and exact graphs' sizeof(T).
template<typename T, typename V>
std::size_t element_bytes(std::size_t n_buffers)
{
  return detail::saturating_mul(n_buffers, sizeof(V)) + 1 + 3 * sizeof(T);
}

/// The three roles' per-element costs, for a problem with `n_buffers`
/// buffer-bearing DAG nodes. No fixed term: every allocation this backend
/// makes is sized off the element count.
template<typename T>
RegionCostModel cost_model_for(std::size_t n_buffers)
{
  RegionCostModel cost;
  cost.sampler_bytes_per_sample = element_bytes<T, T>(n_buffers);
  cost.bound_bytes_per_region = element_bytes<T, cu::interval<T>>(n_buffers);
  cost.exact_bytes_per_region = element_bytes<T, T>(n_buffers);
  cost.fixed_bytes = 0;
  return cost;
}

/// @copydoc cost_model_for(std::size_t)
template<typename T>
RegionCostModel cost_model_for(const dag::Problem<T>& problem)
{
  return cost_model_for<T>(dag::buffer_node_count(problem));
}

}  // namespace cuminlp::backend::graph
