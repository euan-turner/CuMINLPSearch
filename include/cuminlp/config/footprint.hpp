#pragma once

#include <algorithm>
#include <cstddef>

#include "cuminlp/backend/cost_model.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/saturating_arith.hpp"

// The shape-fitting scan config/resolve.hpp's two-phase fit runs, moved out
// of the CUDA-graph backend (design/MODULE_REFACTOR.md §5.5, §11): these
// functions are plain templated C++ over counts and a backend::RegionCostModel,
// with no CUDA type or kernel among them, so a host-only resolver test can
// exercise the exact scan a live backend would run without a GPU. The graph
// backend's own coefficients live in backend/graph/cost.hpp.
namespace cuminlp::config
{

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
    const region::FanOutSpec& fan_out,
    std::size_t sample_points)
{
  std::size_t const integer_width =
      std::max(fan_out.partition_num(), fan_out.enumerate_cap());

  std::size_t widest = 1;  // integers, then continuous, then binaries
  std::size_t discrete = 1;  // integers, then binaries: the enumerable shapes
  std::size_t worst = 0;

  for (std::size_t s = 1; s <= cap; ++s) {
    widest = detail::saturating_mul(widest,
                                    s <= n_integer ? integer_width
                                        : s <= n_integer + n_continuous
                                        ? fan_out.partition_num()
                                        : std::size_t {2});
    // No continuous slot in the first `s` of that ordering means the shape is
    // fully enumerable and pays for the exact graph too.
    bool const enumerable = s <= n_integer || n_continuous == 0;
    worst =
        std::max(worst, cost.bundle_bytes(widest, sample_points, enumerable));

    // The all-discrete shape is a separate maximum rather than a special case
    // of the one above: it is narrower whenever continuous slots exist, but
    // carries the exact graph, so neither dominates the other in general.
    if (s <= n_integer + n_binary) {
      discrete = detail::saturating_mul(
          discrete, s <= n_integer ? integer_width : std::size_t {2});
      worst = std::max(worst, cost.bundle_bytes(discrete, sample_points, true));
    }
  }
  return worst;
}

/**
 * @brief The widest --max-slots whose graphs fit in `budget`, or 0 if
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
                                       const region::FanOutSpec& fan_out,
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

}  // namespace cuminlp::config
