#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/saturating_arith.hpp"

// Split out of graph_replay.cuh: these four functions are plain templated
// C++ over Problem<T>/FanOutSpec, with no CUDA type or kernel among them, but
// graph_replay.cuh also holds real __global__/__device__ kernels and
// <cub/cub.cuh>, so it cannot be #included by a host-only .cpp translation
// unit (a plain host compiler doesn't know what __global__ means). Moving
// them here -- pure code motion, no behaviour change -- lets
// policy_catalogue.hpp's resolver reuse the exact scan graph_replay.cuh's own
// auto-fitting used, from a test target that needs no GPU. graph_replay.cuh
// includes this header for its own use of these functions.
namespace cuminlp::dag
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
std::size_t buffer_node_count(const Problem<T>& problem)
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
    if (problem.graph.nodes[i].op != Op::Const) {
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

/**
 * @brief Device bytes a solve holds at once for one Composition producing
 *        `n_regions` regions.
 *
 * The sum of the graphs GraphDriver caches for that Composition: the point
 * graph at `sample_points` elements per region, the interval graph at one,
 * and the exact graph at one when `enumerable`. All three genuinely coexist
 * -- a Composition takes the exact path only when the box's live variables
 * fit in the slots it filled, so the same Composition is reached both ways
 * on any problem with more variables than slots.
 *
 * Saturating throughout, for the reason composition_fan_out is: with
 * partition_num a runtime value `n_regions` can arrive already at SIZE_MAX,
 * and a wrapped total would read as comfortably in budget.
 */
template<typename T>
std::size_t composition_footprint_bytes(std::size_t n_regions,
                                        std::size_t sample_points,
                                        std::size_t n_buffers,
                                        bool enumerable)
{
  std::size_t const per_point = element_bytes<T, T>(n_buffers);
  std::size_t total = detail::saturating_mul(
      detail::saturating_mul(n_regions, sample_points), per_point);
  total = detail::saturating_add(
      total,
      detail::saturating_mul(n_regions,
                             element_bytes<T, cu::interval<T>>(n_buffers)));
  if (enumerable) {
    total = detail::saturating_add(
        total, detail::saturating_mul(n_regions, per_point));
  }
  return total;
}

/**
 * @brief The widest --max-cycle-size whose graphs fit in `budget`, or 0 if
 *        not even a single slot does.
 *
 * What the CLI uses instead of a hardcoded per-shape constant. The region
 * count is a product over slots, so this is a scan: multiply in one slot's
 * worst-case fan-out at a time and stop at the first cap that doesn't fit.
 *
 * "Worst case" over the whole search, not just the root, on three counts.
 * Slots are charged in the order GreedyCompositionPolicy fills them
 * (binaries, then integers, then continuous), because that order decides
 * which fan-outs a given cap actually buys. Every integer slot is charged
 * max(partition_num, enumerate_cap), since a slot may enumerate or bisect
 * depending on how far its domain has narrowed by the time it is chosen. And
 * the exact graph is charged for any all-binary/all-integer prefix, though a
 * bisecting integer slot would not in fact produce one -- that over-charges
 * by one graph in `sample_points + 2`, and over-charging is the safe
 * direction for a budget.
 *
 * A descendant box never has more live variables than the root, so a full
 * slate of slots at root widths bounds every composition the search reaches.
 *
 * Takes the variable-kind counts and buffer count directly, rather than a
 * `Problem<T>`, so it can be driven from a `PolicyProfile`'s already-computed
 * `ProblemProfile` counts with no `Problem` in hand at all -- that's what
 * makes the resolver a pure function of counts plus calibration. The
 * `Problem<T>` overload below is the convenience wrapper every existing
 * caller keeps using.
 */
template<typename T>
std::size_t auto_max_cycle_size(std::size_t n_binary,
                                std::size_t n_integer,
                                std::size_t n_continuous,
                                std::size_t n_buffers,
                                const FanOutSpec& fan_out,
                                std::size_t sample_points,
                                std::size_t budget,
                                std::size_t ceiling)
{
  std::size_t const integer_width =
      std::max(fan_out.partition_num(), fan_out.enumerate_cap());
  std::size_t const slots =
      std::min(ceiling, n_binary + n_integer + n_continuous);

  std::size_t best = 0;
  std::size_t regions = 1;
  for (std::size_t cap = 1; cap <= slots; ++cap) {
    std::size_t const width = cap <= n_binary ? std::size_t {2}
        : cap <= n_binary + n_integer         ? integer_width
                                              : fan_out.partition_num();
    regions = detail::saturating_mul(regions, width);
    bool const enumerable = cap <= n_binary + n_integer;
    if (composition_footprint_bytes<T>(
            regions, sample_points, n_buffers, enumerable)
        > budget)
    {
      break;
    }
    best = cap;
  }
  return best;
}

/// @copydoc auto_max_cycle_size(std::size_t,std::size_t,std::size_t,std::size_t,const FanOutSpec&,std::size_t,std::size_t,std::size_t)
template<typename T>
std::size_t auto_max_cycle_size(const Problem<T>& problem,
                                const FanOutSpec& fan_out,
                                std::size_t sample_points,
                                std::size_t budget,
                                std::size_t ceiling)
{
  std::size_t n_binary = 0;
  std::size_t n_integer = 0;
  std::size_t n_continuous = 0;
  for (VarKind kind : problem.var_kinds) {
    if (kind == VarKind::Binary) {
      ++n_binary;
    } else if (kind == VarKind::Integer) {
      ++n_integer;
    } else {
      ++n_continuous;
    }
  }
  return auto_max_cycle_size<T>(n_binary,
                                n_integer,
                                n_continuous,
                                buffer_node_count(problem),
                                fan_out,
                                sample_points,
                                budget,
                                ceiling);
}

}  // namespace cuminlp::dag
