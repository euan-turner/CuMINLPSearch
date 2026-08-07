#pragma once

#include <cstddef>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/backend/cost_model.hpp"
#include "cuminlp/model/dag.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/saturating_arith.hpp"

// What the CUDA-graph backend costs, expressed as a backend::RegionCostModel
// (design/MODULE_REFACTOR.md §5.5).
//
// Host-only on purpose, and the reason it is a separate header from
// backend/graph/replay.cuh: the resolver's shape fit runs against these
// coefficients in a test target with no GPU and no CUDA toolchain.
namespace cuminlp::backend::graph
{

/**
 * @brief Number of DAG nodes that will actually get a device buffer.
 *
 * Not simply "every non-Const node": GraphBuilder allocates lazily from the
 * objective and constraint roots (add_expression -> ensure_node), so a node
 * no root reaches never allocates. Const nodes never allocate either -- their
 * payload is consumed by value at the use site (see wire_binary).
 *
 * Counting all non-Const nodes instead would *over*-estimate, and an
 * over-estimate is not the safe direction here: it would make build() refuse
 * configurations that would in fact have fit. A parsed Problem can carry dead
 * nodes that a hand-built one would not.
 *
 * DAGNode ids are topologically ordered (every id in `.in` is < the node's own
 * id), so one reverse sweep suffices -- no recursion or worklist.
 */
template<typename T>
std::size_t buffer_node_count(const model::Problem<T>& problem)
{
  std::size_t const n = problem.graph.nodes.size();
  std::vector<bool> reachable(n, false);

  auto mark = [&](std::size_t id)
  {
    if (id < n) {
      reachable[id] = true;
    }
  };
  mark(problem.objective_root);
  for (const auto& c : problem.constraints) {
    mark(c.root_id);
  }

  std::size_t count = 0;
  for (std::size_t i = n; i-- > 0;) {
    if (!reachable[i]) {
      continue;
    }
    for (std::size_t in_id : problem.graph.nodes[i].in) {
      mark(in_id);
    }
    if (problem.graph.nodes[i].op != model::Op::Const) {
      ++count;
    }
  }
  return count;
}

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
RegionCostModel cost_model_for(const model::Problem<T>& problem)
{
  return cost_model_for<T>(buffer_node_count(problem));
}

/**
 * @brief The bisection budget `B` a device budget affords, closed form
 *        (design/BUDGETED_PARTITION.md §3.2).
 *
 * Under `BisectionBudgetCompositionPolicy`, `N = 2^B` regions is fixed for
 * the *whole solve* -- every composition costs exactly the same, so there is
 * one number to fit rather than config::auto_max_cycle_size's scan over an
 * ordered factorisation `partition_num^k`. `per_region` is what one region's
 * sampler + bounder + enumerator cost together
 * (`RegionCostModel::bundle_bytes` at `n_regions = 1`, `enumerable = true`,
 * since a single-region bundle already includes the enumerator's share) and
 * `B` is the largest power of two that many `per_region`s fit in `budget`.
 *
 * Returns 0 when even one region's three roles do not fit -- the run then
 * proceeds to `GraphReplay::build`'s own out-of-memory report rather than
 * refusing to start, the same fallback `resolve_shape`'s `max_cycle_size`
 * floor uses.
 */
template<typename T>
std::size_t bisection_budget(std::size_t n_buffers,
                             std::size_t sample_points,
                             std::size_t budget)
{
  RegionCostModel const cost = cost_model_for<T>(n_buffers);
  std::size_t const per_region =
      cost.bundle_bytes(1, sample_points, /*enumerable=*/true);
  if (per_region == 0 || budget < per_region) {
    return 0;
  }
  std::size_t ratio = budget / per_region;
  std::size_t b = 0;
  while (ratio > 1) {
    ratio >>= 1;
    ++b;
  }
  return b;
}

/// @copydoc bisection_budget(std::size_t, std::size_t, std::size_t)
template<typename T>
std::size_t bisection_budget(const model::Problem<T>& problem,
                             std::size_t sample_points,
                             std::size_t budget)
{
  return bisection_budget<T>(buffer_node_count(problem), sample_points, budget);
}

}  // namespace cuminlp::backend::graph
