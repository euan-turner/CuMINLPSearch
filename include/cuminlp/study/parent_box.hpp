#pragma once

// The refinement study's parent-box family: a geometric sigma ladder crossed
// with uniformly-placed sub-boxes of the root
// (design/REFINEMENT_STUDY.md §3.1).
//
// Host-only and free of CUDA kernels -- it manipulates cu::interval as a plain
// aggregate, so every property in §3.1 (containment, reproducibility, discrete
// snapping) is testable without a GPU.
//
// Key external dependencies: cu::interval (as a value type only), and
// model::VarKind to tell continuous from discrete variables. Companion doc:
// design/REFINEMENT_STUDY.md.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/model/problem.hpp"

namespace cuminlp::study
{

/// `1, 1/2, 1/4, ..., 1/2^rungs` -- `rungs + 1` values, widest first.
///
/// Widest first so that a run truncated part-way still holds the rungs whose
/// boxes are largest, which are the ones the `N` ladder can least afford to
/// re-measure later.
inline std::vector<double> sigma_ladder(std::size_t rungs)
{
  std::vector<double> out;
  out.reserve(rungs + 1);
  double sigma = 1.0;
  for (std::size_t i = 0; i <= rungs; ++i) {
    out.push_back(sigma);
    sigma *= 0.5;
  }
  return out;
}

/// How many placements a rung admits. `sigma == 1` leaves no freedom -- the
/// only box of relative width 1 inside the root is the root -- so drawing
/// `reps` copies of it would be `reps - 1` duplicate launches reported as
/// independent samples (§3.1).
inline std::size_t placements_for(double sigma, std::size_t reps)
{
  return (sigma >= 1.0) ? 1 : reps;
}

/**
 * @brief One parent box of relative width `sigma`, placed uniformly at random
 *        inside `root`.
 *
 * Per variable: width `sigma * w_i`, centre drawn uniformly from the range
 * that keeps the shrunken box inside the root. A variable already fixed in
 * the root (`w_i == 0`) stays fixed.
 *
 * **Discrete variables** are snapped *outward* to the integer lattice after
 * shrinking, then clamped back inside the root (§3.1). Outward, because a box
 * whose endpoints are not integers does not correspond to any sub-problem the
 * discrete structure can occupy; clamped, because outward snapping at a root
 * boundary would otherwise leave the root. A sigma small enough that no
 * integer survives collapses the variable to a single lattice point rather
 * than producing an empty interval.
 *
 * Consequence, which the caller must report rather than hide: `sigma` is
 * nominal for discrete variables. Use `achieved_widths` for what was actually
 * measured.
 *
 * `sigma >= 1` returns `root` exactly, consuming no randomness.
 */
template<typename T>
std::vector<cu::interval<T>> draw_parent_box(
    std::span<const cu::interval<T>> root,
    std::span<const model::VarKind> kinds,
    double sigma,
    std::mt19937_64& rng)
{
  std::vector<cu::interval<T>> box(root.begin(), root.end());
  if (sigma >= 1.0) {
    return box;
  }

  for (std::size_t v = 0; v < root.size(); ++v) {
    T const lo = root[v].lb;
    T const hi = root[v].ub;
    T const width = hi - lo;
    if (!(width > 0)) {
      continue;  // already fixed, or degenerate; leave it alone
    }

    T const half = static_cast<T>(sigma * static_cast<double>(width) * 0.5);
    // Drawn even when the resulting range is a single point, so that the
    // number of variates consumed depends only on (root, sigma) and not on
    // the values drawn -- otherwise the same seed would produce different
    // boxes depending on rounding.
    std::uniform_real_distribution<double> d(static_cast<double>(lo + half),
                                             static_cast<double>(hi - half));
    T const centre = static_cast<T>(d(rng));

    T lb = centre - half;
    T ub = centre + half;

    if (v < kinds.size() && kinds[v] != model::VarKind::Continuous) {
      lb = std::floor(lb);
      ub = std::ceil(ub);
      if (ub < lb) {
        ub = lb;
      }
    }

    // Never leave the root, whatever the shrink and snap did.
    lb = std::max(lb, lo);
    ub = std::min(ub, hi);
    if (ub < lb) {
      ub = lb;
    }

    box[v] = cu::interval<T> {lb, ub};
  }
  return box;
}

/**
 * @brief One parent box of fixed *absolute* width `width` per variable,
 *        placed uniformly at random inside `root` (design/REFINEMENT_STUDY.md
 *        §7.3).
 *
 * Unlike `draw_parent_box`, the width is not relative to each variable's own
 * root width -- every live variable gets the same absolute width, so a
 * problem whose declared bounds are a huge default (§7.2's `±1,000,000`
 * finding) and one whose bounds are already tight are placed on equal
 * footing. This is the point: it isolates the theorem's `K` (the objective's
 * local behaviour) from `w(X)` (how wide the root box happened to be
 * declared), which a sigma-relative ladder cannot do since `sigma = 1` always
 * means "the whole root" regardless of how large that is.
 *
 * `width >= root width_i` clamps to the root's own width in that dimension
 * (the same fallback `sigma >= 1` uses) rather than growing past the root --
 * there is no sub-box wider than the root to place. Fixed variables
 * (`w_i == 0`) stay fixed. Discrete snapping is identical to
 * `draw_parent_box`.
 *
 * `center_range` bounds where the centre is drawn from: the placement window
 * `[lo + half, hi - half]` is intersected with `[-center_range,
 * center_range]` before the centre is sampled. Default +inf, i.e. no
 * restriction -- placement covers the whole root, which for a variable whose
 * bound was never declared and defaulted to `±1,000,000` (§7.2's finding)
 * means almost every draw lands far out in that synthetic padding rather
 * than near where the model's declared bound actually was. A finite
 * `center_range` (e.g. `10`) keeps placements near the origin regardless of
 * how wide the root happens to be, which is what makes a fixed-width
 * comparison across instances with and without default-bound padding
 * meaningful in the first place (§7.3).
 */
template<typename T>
std::vector<cu::interval<T>> draw_fixed_width_box(
    std::span<const cu::interval<T>> root,
    std::span<const model::VarKind> kinds,
    double width,
    std::mt19937_64& rng,
    double center_range = std::numeric_limits<double>::infinity())
{
  std::vector<cu::interval<T>> box(root.begin(), root.end());

  for (std::size_t v = 0; v < root.size(); ++v) {
    T const lo = root[v].lb;
    T const hi = root[v].ub;
    T const root_width = hi - lo;
    if (!(root_width > 0)) {
      continue;  // already fixed, or degenerate; leave it alone
    }

    double const clamped =
        std::min(width, static_cast<double>(root_width));
    T const half = static_cast<T>(clamped * 0.5);
    double const window_lo =
        std::max(static_cast<double>(lo + half), -center_range);
    double const window_hi =
        std::min(static_cast<double>(hi - half), center_range);
    // Drawn even when clamped == root_width or the window collapses to a
    // point, same reasoning as draw_parent_box: variate consumption depends
    // only on (root, width, center_range), never on the values drawn.
    std::uniform_real_distribution<double> d(std::min(window_lo, window_hi),
                                             std::max(window_lo, window_hi));
    T const centre = static_cast<T>(d(rng));

    T lb = centre - half;
    T ub = centre + half;

    if (v < kinds.size() && kinds[v] != model::VarKind::Continuous) {
      lb = std::floor(lb);
      ub = std::ceil(ub);
      if (ub < lb) {
        ub = lb;
      }
    }

    lb = std::max(lb, lo);
    ub = std::min(ub, hi);
    if (ub < lb) {
      ub = lb;
    }

    box[v] = cu::interval<T> {lb, ub};
  }
  return box;
}

/// Per-variable widths of `box` -- what was actually achieved, as opposed to
/// the `sigma` that was requested (§3.1, §6.5).
template<typename T>
std::vector<T> achieved_widths(std::span<const cu::interval<T>> box)
{
  std::vector<T> out;
  out.reserve(box.size());
  for (const auto& b : box) {
    out.push_back(b.ub - b.lb);
  }
  return out;
}

/// The mean of `width_i(box) / width_i(root)` over variables live in the root.
/// The single number a row can carry to say how far the achieved box drifted
/// from its nominal `sigma`; 1.0 when `box == root`.
template<typename T>
double achieved_relative_width(std::span<const cu::interval<T>> box,
                               std::span<const cu::interval<T>> root)
{
  double total = 0.0;
  std::size_t live = 0;
  for (std::size_t v = 0; v < root.size() && v < box.size(); ++v) {
    T const rw = root[v].ub - root[v].lb;
    if (!(rw > 0)) {
      continue;
    }
    total += static_cast<double>(box[v].ub - box[v].lb)
        / static_cast<double>(rw);
    ++live;
  }
  return (live == 0) ? 1.0 : total / static_cast<double>(live);
}

/// True iff every component of `box` lies inside `root`. The invariant every
/// drawn parent box must satisfy; cheap enough to assert per row.
template<typename T>
bool inside(std::span<const cu::interval<T>> box,
            std::span<const cu::interval<T>> root)
{
  if (box.size() != root.size()) {
    return false;
  }
  for (std::size_t v = 0; v < box.size(); ++v) {
    if (box[v].lb < root[v].lb || box[v].ub > root[v].ub
        || box[v].ub < box[v].lb)
    {
      return false;
    }
  }
  return true;
}

}  // namespace cuminlp::study
