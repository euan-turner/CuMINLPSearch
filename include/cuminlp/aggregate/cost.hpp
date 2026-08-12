#pragma once

#include <cstddef>

#include <cuinterval/interval.h>

#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/saturating_arith.hpp"

// What the aggregate backend costs, and how its device budget divides between
// the bounder and the sampler (design/AGGREGATE_BOUNDING.md §6.3).
//
// Host-only, and a separate header from backend/aggregate/replay.cuh for the
// same reason backend/graph/cost.hpp is separate from replay.cuh: the fit runs
// in a test target with no GPU and no CUDA toolchain, and the numbers it
// produces go into the run's PARAMS line, which has to be explainable without
// launching anything.
namespace cuminlp::aggregate
{

/**
 * @brief Bytes one subregion costs the bounder.
 *
 * One `cu::interval<T>` per buffer-bearing DAG node, plus `feasible[]` and
 * four `T` arrays: `obj_lb`, `obj_ub`, `masked_lb`, `masked_ub`.
 *
 * One `T` more per element than `graph::element_bytes<T, cu::interval<T>>`,
 * which counts three. This backend reduces the *lower* bound as well as the
 * upper, so it masks both and carries a second masked array. The `k` outputs
 * and the CUB scratch are `O(k)` and `O(sqrt(M))`, not `O(N)`, so they do not
 * appear here.
 */
template<typename T>
std::size_t bounder_element_bytes(std::size_t n_buffers)
{
  return detail::saturating_mul(n_buffers, sizeof(cu::interval<T>)) + 1
      + 4 * sizeof(T);
}

/// Bytes one sampled point costs. The existing point graph, unchanged, so
/// this is `graph::element_bytes<T, T>` by definition rather than by
/// coincidence -- the sampler *is* `graph::PointGraphReplay`.
template<typename T>
std::size_t sampler_element_bytes(std::size_t n_buffers)
{
  return backend::graph::element_bytes<T, T>(n_buffers);
}

/// How the device budget divides (§6.3).
struct AggregateBudget
{
  std::size_t bounder_bytes = 0;
  std::size_t sampler_bytes = 0;
};

/// The default share of the device budget given to the sampler (§6.3).
/// Small in budget yet generous in points, because a point element is roughly
/// half an interval element: at this fraction the sampler still draws on the
/// order of `0.2 * N` points.
inline constexpr double default_sample_budget_fraction = 0.10;

/// Split `total` between the two graphs. `fraction` is the sampler's share,
/// clamped to `[0, 1)` -- a sampler given the whole budget would leave the
/// bounder unable to build at all, which is a configuration error rather than
/// a tradeoff.
inline AggregateBudget split_device_budget(std::size_t total, double fraction)
{
  double const f = fraction < 0.0 ? 0.0 : (fraction > 0.9 ? 0.9 : fraction);
  auto const sampler = static_cast<std::size_t>(static_cast<double>(total) * f);
  return AggregateBudget {total - sampler, sampler};
}

/**
 * @brief The largest power of two `N` whose per-element cost fits `budget`.
 *
 * The aggregate counterpart of `graph::bisection_budget`, and deliberately
 * the same shape: a power of two, so `M = N / k` stays exact and the refine's
 * bisection budget stays an integer.
 *
 * Returns 0 when not even one element fits, leaving the caller to fail
 * through the backend's own out-of-memory report rather than refusing to
 * start.
 */
inline std::size_t fit_power_of_two(std::size_t element_bytes,
                                    std::size_t budget)
{
  if (element_bytes == 0 || budget < element_bytes) {
    return 0;
  }
  std::size_t ratio = budget / element_bytes;
  std::size_t n = 1;
  while (ratio > 1) {
    ratio >>= 1;
    n <<= 1;
  }
  return n;
}

}  // namespace cuminlp::aggregate
