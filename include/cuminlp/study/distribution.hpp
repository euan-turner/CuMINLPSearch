#pragma once

// The refinement study's answer to Q2: is a set of per-subregion bounds all
// alike, or are its extremes isolated outliers?
// (design/REFINEMENT_STUDY.md §2.2, layers 1-5.)
//
// The question is deliberately not "what is the spread". A quantile ladder
// reports a smooth-looking spread over a distribution that is actually a few
// tied blocks, and on this workload heavy exact ties are expected rather than
// pathological: lb_r is constant across every subregion whose differing slots
// feed variables the objective does not depend on. So degeneracy is measured
// first, and everything downstream tolerates a zero dispersion.
//
// Pure host arithmetic over a span of values plus a feasibility mask -- no
// CUDA, no backend types, so §2.2's claims are testable on planted arrays
// (design/REFINEMENT_STUDY.md §7, stage 4).
//
// Companion doc: design/REFINEMENT_STUDY.md.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace cuminlp::study
{

/// §2.2 layer 2's tail-dense ladder. Dense at both ends because both tails
/// are interesting, and for different reasons (§2.3); sparse in the middle,
/// where nothing this study cares about happens.
inline constexpr double kQuantileLevels[] = {0.0,
                                             0.001,
                                             0.005,
                                             0.01,
                                             0.02,
                                             0.05,
                                             0.10,
                                             0.25,
                                             0.50,
                                             0.75,
                                             0.90,
                                             0.95,
                                             0.98,
                                             0.99,
                                             0.995,
                                             0.999,
                                             1.0};

inline constexpr std::size_t kQuantileCount =
    sizeof(kQuantileLevels) / sizeof(kQuantileLevels[0]);

/// Order statistics `k` at which layer 4 measures how isolated an extreme is.
inline constexpr std::size_t kIsolationRanks[] = {2, 10, 100};

inline constexpr std::size_t kIsolationCount =
    sizeof(kIsolationRanks) / sizeof(kIsolationRanks[0]);

/**
 * @brief §2.2's five-layer summary of one distribution of bounds.
 *
 * Every field that divides by a dispersion measure is optional and is left
 * empty when `iqr` is zero or non-finite. That case is a *result* -- "these
 * bounds are all alike" -- and emitting `inf` or `nan` for it would let it
 * average into a summary as though it were a measurement.
 */
struct Distribution
{
  std::size_t count = 0;  ///< values summarised (unexcluded only)

  // -- layer 1: degeneracy. Are they all the same? ------------------------
  double distinct_frac = 0.0;  ///< distinct values / count
  double modal_frac = 0.0;  ///< share taking the most common value
  double iqr = 0.0;  ///< q75 - q25; legitimately 0

  // -- layer 2: tail-dense quantiles --------------------------------------
  std::vector<double> quantiles;  ///< parallel to kQuantileLevels
  std::optional<double> mean;  ///< empty if any value is non-finite
  std::optional<double> sd;
  double nonfinite_frac = 0.0;  ///< share that is +/-inf or NaN

  // -- layer 3: Tukey outlier scores, per tail ----------------------------
  std::optional<double> low_score;  ///< (q25 - min) / iqr
  std::optional<double> high_score;  ///< (max - q75) / iqr
  double low_frac = 0.0;  ///< mass below q25 - 1.5*iqr
  double high_frac = 0.0;  ///< mass above q75 + 1.5*iqr

  // -- layer 4: isolation of the extremes ---------------------------------
  /// Parallel to kIsolationRanks. `low_gap[i]` is the distance from the
  /// minimum to the `kIsolationRanks[i]`-th smallest value, in IQR units.
  /// Empty where the count is too small for that rank, or the IQR is zero.
  ///
  /// This is what a fraction cannot express: at N = 2^20 a lone minimum and a
  /// cluster of fifty both round to 0.00, but a large `low_gap[0]` says the
  /// minimum stands alone while a small `low_gap[0]` with a large `low_gap[2]`
  /// says a small cluster does.
  std::vector<std::optional<double>> low_gap;
  std::vector<std::optional<double>> high_gap;

  // -- shape, for recovering the picture without N doubles ----------------
  std::vector<std::size_t> histogram;  ///< linear bins over [min, max]

  bool empty() const { return count == 0; }
};

namespace detail
{

/// Linear interpolation between order statistics -- numpy/pandas' default
/// ("type 7"), chosen so a reader can reproduce these numbers with the tool
/// they will actually plot them in. `sorted` must be non-empty.
inline double quantile_of(std::span<const double> sorted, double q)
{
  if (sorted.size() == 1) {
    return sorted[0];
  }
  double const pos = q * static_cast<double>(sorted.size() - 1);
  double const lo = std::floor(pos);
  double const hi = std::ceil(pos);
  auto const i = static_cast<std::size_t>(lo);
  auto const j = static_cast<std::size_t>(hi);
  if (i == j) {
    return sorted[i];
  }
  double const frac = pos - lo;
  return sorted[i] * (1.0 - frac) + sorted[j] * frac;
}

}  // namespace detail

/**
 * @brief Summarise the unexcluded entries of `values` (§2.2).
 *
 * `feasible` may be empty, in which case every entry is summarised -- that is
 * the unmasked view §2.5 also wants.
 *
 * `bins` is the histogram resolution; 0 suppresses the histogram.
 */
template<typename T>
Distribution summarise(std::span<const T> values,
                       std::span<const unsigned char> feasible,
                       std::size_t bins = 64)
{
  Distribution d;

  std::vector<double> v;
  v.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!feasible.empty() && (i >= feasible.size() || feasible[i] == 0)) {
      continue;
    }
    v.push_back(static_cast<double>(values[i]));
  }
  if (v.empty()) {
    return d;
  }

  std::sort(v.begin(), v.end());
  d.count = v.size();
  auto const n = static_cast<double>(d.count);

  // -- layer 1 -------------------------------------------------------------
  std::size_t distinct = 1;
  std::size_t run = 1;
  std::size_t best_run = 1;
  for (std::size_t i = 1; i < v.size(); ++i) {
    // `v` is sorted, so a strict `<` is exactly "distinct from its
    // predecessor" -- and avoids an equality comparison on doubles, which
    // this project builds with -Werror=float-equal.
    if (v[i - 1] < v[i]) {
      ++distinct;
      run = 1;
    } else {
      ++run;
      best_run = std::max(best_run, run);
    }
  }
  d.distinct_frac = static_cast<double>(distinct) / n;
  d.modal_frac = static_cast<double>(best_run) / n;

  // -- layer 2 -------------------------------------------------------------
  d.quantiles.reserve(kQuantileCount);
  for (double q : kQuantileLevels) {
    d.quantiles.push_back(detail::quantile_of(v, q));
  }

  std::size_t nonfinite = 0;
  for (double x : v) {
    nonfinite += std::isfinite(x) ? 0 : 1;
  }
  d.nonfinite_frac = static_cast<double>(nonfinite) / n;
  if (nonfinite == 0) {
    double sum = 0.0;
    for (double x : v) {
      sum += x;
    }
    double const mean = sum / n;
    double var = 0.0;
    for (double x : v) {
      double const dx = x - mean;
      var += dx * dx;
    }
    d.mean = mean;
    d.sd = std::sqrt(var / n);
  }

  double const q25 = detail::quantile_of(v, 0.25);
  double const q75 = detail::quantile_of(v, 0.75);
  double const lo = v.front();
  double const hi = v.back();
  d.iqr = q75 - q25;

  // The zero-dispersion gate for layers 3 and 4. Non-finite as well as zero:
  // an unbounded relaxation gives an infinite IQR, against which every
  // distance is 0 and every conclusion drawn would be spurious.
  bool const has_scale = d.iqr > 0.0 && std::isfinite(d.iqr);

  // -- layer 3 -------------------------------------------------------------
  if (has_scale) {
    d.low_score = (q25 - lo) / d.iqr;
    d.high_score = (hi - q75) / d.iqr;

    double const low_fence = q25 - 1.5 * d.iqr;
    double const high_fence = q75 + 1.5 * d.iqr;
    std::size_t below = 0;
    std::size_t above = 0;
    for (double x : v) {
      below += (x < low_fence) ? 1 : 0;
      above += (x > high_fence) ? 1 : 0;
    }
    d.low_frac = static_cast<double>(below) / n;
    d.high_frac = static_cast<double>(above) / n;
  }

  // -- layer 4 -------------------------------------------------------------
  d.low_gap.assign(kIsolationCount, std::nullopt);
  d.high_gap.assign(kIsolationCount, std::nullopt);
  if (has_scale) {
    for (std::size_t i = 0; i < kIsolationCount; ++i) {
      std::size_t const k = kIsolationRanks[i];
      if (d.count < k) {
        continue;
      }
      d.low_gap[i] = (v[k - 1] - v.front()) / d.iqr;
      d.high_gap[i] = (v.back() - v[d.count - k]) / d.iqr;
    }
  }

  // -- histogram -----------------------------------------------------------
  if (bins > 0 && std::isfinite(lo) && std::isfinite(hi)) {
    d.histogram.assign(bins, 0);
    double const span = hi - lo;
    for (double x : v) {
      std::size_t idx = 0;
      if (span > 0) {
        auto const scaled = (x - lo) / span * static_cast<double>(bins);
        idx = static_cast<std::size_t>(scaled);
        idx = std::min(idx, bins - 1);
      }
      ++d.histogram[idx];
    }
  }

  return d;
}

/**
 * @brief §2.2 layer 5: the share of subregions whose lower bound sits within
 *        `eps` of the hull's floor.
 *
 * The interpretation key for Q1 (§2.1). A tiny `c(0.01)` says `L_N` is set by
 * a handful of subregions and the rest are prunable -- subdivision is buying
 * discrimination. A `c(0.25)` near 1 says every subregion looks equally good
 * to the relaxation, and a tighter width ratio overstates the search benefit.
 *
 * Measured against the hull's *full* span `[hull_lb, hull_ub]` rather than the
 * range of `lb` alone, so that it is comparable across rows: the width of the
 * lb distribution is itself one of the things under study.
 */
template<typename T>
double concentration(std::span<const T> lb,
                     std::span<const unsigned char> feasible,
                     T hull_lb,
                     T hull_ub,
                     double eps)
{
  double const span = static_cast<double>(hull_ub) - static_cast<double>(hull_lb);
  if (!(span > 0) || !std::isfinite(span)) {
    return 0.0;
  }
  double const threshold = static_cast<double>(hull_lb) + eps * span;

  std::size_t within = 0;
  std::size_t total = 0;
  for (std::size_t i = 0; i < lb.size(); ++i) {
    if (!feasible.empty() && (i >= feasible.size() || feasible[i] == 0)) {
      continue;
    }
    ++total;
    within += (static_cast<double>(lb[i]) <= threshold) ? 1 : 0;
  }
  return (total == 0) ? 0.0
                      : static_cast<double>(within) / static_cast<double>(total);
}

}  // namespace cuminlp::study
