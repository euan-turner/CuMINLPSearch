#pragma once

// The refinement study's pure host arithmetic: hull reductions over retained
// per-subregion bounds, and the Q1 gain metrics derived from them
// (design/REFINEMENT_STUDY.md §2.1, §2.5).
//
// Deliberately free of CUDA and of any backend type -- it takes spans of
// doubles and a feasibility mask, so every claim in §2 is testable on planted
// arrays with no GPU and no solve (§7, stages 2 and 4). The only header it
// needs is <span>; that is the point.
//
// Depends on: nothing in this project. Companion doc:
// design/REFINEMENT_STUDY.md.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>

namespace cuminlp::study
{

/// An objective range over some set of subregions. `any` is false when the
/// set was empty -- for a masked hull, that means every subregion was
/// excluded, which is a result (§2.1) rather than an error, so it is
/// represented rather than signalled.
template<typename T>
struct Hull
{
  T lb = std::numeric_limits<T>::infinity();
  T ub = -std::numeric_limits<T>::infinity();
  bool any = false;

  /// `+inf` for an empty hull, so that a caller comparing widths does not
  /// have to special-case it into a false "perfectly tight" reading.
  T width() const
  {
    return any ? ub - lb : std::numeric_limits<T>::infinity();
  }
};

/**
 * @brief The interval hull over every subregion, ignoring feasibility.
 *
 * §2.5's unmasked hull: what refinement alone buys, with no contribution from
 * constraint propagation excluding subregions.
 */
template<typename T>
Hull<T> unmasked_hull(std::span<const T> lb, std::span<const T> ub)
{
  Hull<T> h;
  std::size_t const n = std::min(lb.size(), ub.size());
  for (std::size_t i = 0; i < n; ++i) {
    h.any = true;
    h.lb = std::min(h.lb, lb[i]);
    h.ub = std::max(h.ub, ub[i]);
  }
  return h;
}

/**
 * @brief The interval hull over unexcluded subregions only.
 *
 * §2.5's masked hull, and the sound one for search -- it is exactly the
 * reduction `AggregateBound` performs, which stage 1's test pins bitwise.
 * Its difference from `unmasked_hull` is the constraint contribution.
 *
 * `feasible[r] != 0` means region `r`'s relaxation did not *prove* it
 * infeasible. It does not mean `r` contains a feasible point.
 */
template<typename T>
Hull<T> masked_hull(std::span<const T> lb,
                    std::span<const T> ub,
                    std::span<const unsigned char> feasible)
{
  Hull<T> h;
  std::size_t const n = std::min({lb.size(), ub.size(), feasible.size()});
  for (std::size_t i = 0; i < n; ++i) {
    if (feasible[i] == 0) {
      continue;
    }
    h.any = true;
    h.lb = std::min(h.lb, lb[i]);
    h.ub = std::max(h.ub, ub[i]);
  }
  return h;
}

/**
 * @brief `min_r ub_r` over unexcluded subregions -- see §2.4's caveat.
 *
 * A sound upper bound on the objective's minimum over the *box*, obtained
 * with no sampling. It is **not** a sound primal bound when the problem has
 * constraints, because an unexcluded subregion is not known to contain a
 * feasible point, and it must never be folded into an incumbent. Reported
 * only.
 *
 * Empty when every subregion was excluded.
 */
template<typename T>
std::optional<T> min_upper_bound(std::span<const T> ub,
                                 std::span<const unsigned char> feasible)
{
  std::optional<T> best;
  std::size_t const n = std::min(ub.size(), feasible.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (feasible[i] == 0) {
      continue;
    }
    best = best ? std::min(*best, ub[i]) : ub[i];
  }
  return best;
}

/// Share of subregions the relaxation proved infeasible.
inline double excluded_fraction(std::span<const unsigned char> feasible)
{
  if (feasible.empty()) {
    return 0.0;
  }
  std::size_t live = 0;
  for (unsigned char f : feasible) {
    live += (f != 0) ? 1 : 0;
  }
  return 1.0 - static_cast<double>(live) / static_cast<double>(feasible.size());
}

/**
 * @brief §2.1's three Q1 metrics, comparing a refined hull to the baseline.
 *
 * `dual_gain` is the headline: the only one that changes what a search can
 * prune. `primal_side_gain` is diagnostic -- `U_N` is attained at no known
 * point and is not an incumbent improvement.
 *
 * `width_ratio` is empty rather than NaN when the baseline has zero width
 * (a constant objective on the box) or is unbounded: both are real outcomes
 * that a ratio cannot express, and emitting NaN would let them average into
 * a summary silently.
 */
template<typename T>
struct Gains
{
  T dual_gain = 0;  ///< L_N - L0, non-negative
  T primal_side_gain = 0;  ///< U0 - U_N, non-negative
  std::optional<double> width_ratio;  ///< (U_N - L_N) / (U0 - L0), in (0, 1]
};

template<typename T>
Gains<T> gains(const Hull<T>& baseline, const Hull<T>& refined)
{
  Gains<T> g;
  if (!baseline.any || !refined.any) {
    return g;
  }
  g.dual_gain = refined.lb - baseline.lb;
  g.primal_side_gain = baseline.ub - refined.ub;

  T const base_width = baseline.ub - baseline.lb;
  if (base_width > 0 && std::isfinite(base_width)
      && std::isfinite(refined.ub - refined.lb))
  {
    g.width_ratio = static_cast<double>(refined.ub - refined.lb)
        / static_cast<double>(base_width);
  }
  return g;
}

}  // namespace cuminlp::study
