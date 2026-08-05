#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cuminlp/backend/cost_model.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/saturating_arith.hpp"

// Split out of graph_replay.cuh: these functions are plain templated C++ over
// Problem<T>/FanOutSpec, with no CUDA type or kernel among them, but
// graph_replay.cuh also holds real __global__/__device__ kernels and
// <cub/cub.cuh>, so it cannot be #included by a host-only .cpp translation
// unit (a plain host compiler doesn't know what __global__ means). Moving
// them here lets policy_catalogue.hpp's resolver reuse the exact scan
// graph_replay.cuh's own auto-fitting used, from a test target that needs no
// GPU.
//
// The shape enumeration below is now generic over a
// backend::RegionCostModel rather than over a value type and a buffer count
// (design/MODULE_REFACTOR.md §5.5): what a region costs is the backend's
// answer, and asking for it as four coefficients is what lets this scan run
// against hand-written numbers in a host-only test. The graph backend's own
// coefficients live in backend/graph/cost.hpp.
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

/**
 * @brief The most device memory any Composition of at most `cap` slots can
 *        cost, over every box the search could ever reach.
 *
 * The quantity a cap has to be fitted against, and the reason this is not
 * simply `cost.bundle_bytes` of one shape: with `cap` slots and a
 * problem of these kind counts there are many reachable Compositions, they
 * differ by orders of magnitude in region count, and the *root's* is not the
 * widest.
 *
 * That last point is what this function exists to get right. Greedy fills
 * binaries first, so on a problem with plenty of binaries the root
 * composition is all-binary at fan-out 2 -- but binaries *resolve* as the
 * search descends, and a descendant with only a few live binaries left fills
 * the freed slots with continuous variables at `partition_num` each. On
 * batch.gms (24 binary, 22 continuous) a cap of 14 costs 2^14 = 16k regions
 * at the root and 2^10 x 64^4 = 17.2e9 regions ten levels down: the same cap,
 * four orders of magnitude apart, and only the second one is the budget.
 *
 * So slots are charged in *descending width* order, not in the order the
 * policy fills them. Two candidate shapes bound every reachable composition
 * between them:
 *
 *  - the widest shape outright -- integers at max(partition_num,
 *    enumerate_cap), then continuous at partition_num, then binaries at 2,
 *    each up to how many variables of that kind the problem has. Widths are
 *    ordered integer >= continuous >= binary always (partition_num >= 2 is a
 *    FanOutSpec invariant), so filling greedily by width maximises the
 *    product;
 *  - the widest *enumerable* shape -- integers then binaries, no continuous
 *    slot -- which pays for a third role (the enumerator BackendCache holds
 *    for a fully-enumerable Composition) that the first candidate may not.
 *
 * Every integer slot is charged max(partition_num, enumerate_cap), since a
 * slot may enumerate or bisect depending on how far its domain has narrowed
 * by the time it is chosen; that over-charges a mixed integer run, in the
 * safe direction.
 *
 * Monotone in `cap` by construction (it is a running maximum over shapes of
 * every width up to `cap`), which is what lets the scan below stop at the
 * first cap that doesn't fit.
 */
inline std::size_t worst_composition_footprint(
    std::size_t cap,
    std::size_t n_binary,
    std::size_t n_integer,
    std::size_t n_continuous,
    const backend::RegionCostModel& cost,
    const FanOutSpec& fan_out,
    std::size_t sample_points)
{
  std::size_t const integer_width =
      std::max(fan_out.partition_num(), fan_out.enumerate_cap());

  std::size_t widest = 1;  // integers, then continuous, then binaries
  std::size_t discrete = 1;  // integers, then binaries: the enumerable shapes
  std::size_t worst = 0;

  for (std::size_t s = 1; s <= cap; ++s) {
    widest = detail::saturating_mul(
        widest,
        s <= n_integer                    ? integer_width
            : s <= n_integer + n_continuous ? fan_out.partition_num()
                                            : std::size_t {2});
    // No continuous slot in the first `s` of that ordering means the shape is
    // fully enumerable and pays for the exact graph too.
    bool const enumerable = s <= n_integer || n_continuous == 0;
    worst = std::max(
        worst, cost.bundle_bytes(widest, sample_points, enumerable));

    // The all-discrete shape is a separate maximum rather than a special case
    // of the one above: it is narrower whenever continuous slots exist, but
    // carries the exact graph, so neither dominates the other in general.
    if (s <= n_integer + n_binary) {
      discrete = detail::saturating_mul(
          discrete, s <= n_integer ? integer_width : std::size_t {2});
      worst = std::max(worst,
                       cost.bundle_bytes(discrete, sample_points, true));
    }
  }
  return worst;
}

/**
 * @brief The widest --max-cycle-size whose graphs fit in `budget`, or 0 if
 *        not even a single slot does.
 *
 * What the CLI uses instead of a hardcoded per-shape constant. A scan, since
 * `worst_composition_footprint` is monotone in the cap: stop at the first cap
 * whose worst reachable composition doesn't fit.
 *
 * The cap it returns holds for *every* box the search reaches, not just the
 * root -- see worst_composition_footprint for why those differ and why the
 * root is not the binding case.
 *
 * Takes the variable-kind counts and a cost model directly, rather than a
 * `Problem<T>`, so it can be driven from a `ProblemProfile`'s already-computed
 * counts with no `Problem` and no backend instance in hand at all -- that's
 * what makes the resolver a pure function of counts, calibration and the
 * backend's four coefficients.
 */
inline std::size_t auto_max_cycle_size(std::size_t n_binary,
                                       std::size_t n_integer,
                                       std::size_t n_continuous,
                                       const backend::RegionCostModel& cost,
                                       const FanOutSpec& fan_out,
                                       std::size_t sample_points,
                                       std::size_t budget,
                                       std::size_t ceiling)
{
  std::size_t const slots =
      std::min(ceiling, n_binary + n_integer + n_continuous);

  std::size_t best = 0;
  for (std::size_t cap = 1; cap <= slots; ++cap) {
    if (worst_composition_footprint(cap,
                                    n_binary,
                                    n_integer,
                                    n_continuous,
                                    cost,
                                    fan_out,
                                    sample_points)
        > budget)
    {
      break;
    }
    best = cap;
  }
  return best;
}

}  // namespace cuminlp::dag
