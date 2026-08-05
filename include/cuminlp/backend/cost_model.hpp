#pragma once

#include <cstddef>

#include "cuminlp/saturating_arith.hpp"

// The device footprint of a backend, as a value rather than as an object
// (design/MODULE_REFACTOR.md §5.5).
//
// The resolver has to fit a search shape against a backend's footprint before
// any backend instance -- or any device -- exists, and it must do so in a
// host-only test target. So a backend publishes four coefficients computed
// from the Problem, and everything above it does arithmetic on those instead
// of on the backend's types. `config`'s shape fit and `explain_over_budget`
// are both written against this struct and nothing else, which is what makes
// them testable against hand-written coefficients with no GPU in sight.
//
// Stated assumption: a backend's footprint is affine in the region count.
// True of the CUDA-graph backend (one buffer per reachable DAG node, plus
// four per-element arrays) and of a fused JIT backend (fewer coefficients,
// same form). A backend whose footprint is not affine needs a virtual cost
// model instead of this value -- §15 records that as an open item.
namespace cuminlp::backend
{

struct RegionCostModel
{
  std::size_t sampler_bytes_per_sample = 0;  ///< sampler, per drawn point
  std::size_t bound_bytes_per_region = 0;  ///< bounder, per region
  std::size_t exact_bytes_per_region = 0;  ///< enumerator, per region
  std::size_t fixed_bytes = 0;  ///< per instance, size-independent

  bool operator==(const RegionCostModel&) const = default;

  /// Device bytes the one solve-lifetime sampler holds (§5.2).
  std::size_t sampler_bytes(std::size_t n_samples) const
  {
    return detail::saturating_add(
        detail::saturating_mul(n_samples, sampler_bytes_per_sample),
        fixed_bytes);
  }

  /**
   * @brief Device bytes one Composition's roles hold at once.
   *
   * All three genuinely coexist: a Composition takes the exact path only when
   * the box's live variables fit in the slots it filled, so the same
   * Composition is reached both ways on any problem with more variables than
   * slots.
   *
   * Saturating throughout, for the reason composition_fan_out is: with
   * partition_num a runtime value `n_regions` can arrive already at SIZE_MAX,
   * and a wrapped total would read as comfortably in budget.
   *
   * `samples_per_region` is here only while the sampler is still
   * composition-coupled. §5.2 moves it out of the bundle and into
   * `sampler_bytes` once per solve, at which point this drops to the
   * `(n_regions, enumerable)` pair §5.5 specifies and the largest of the
   * three terms stops scaling with the composition.
   */
  std::size_t bundle_bytes(std::size_t n_regions,
                           std::size_t samples_per_region,
                           bool enumerable) const
  {
    std::size_t total = detail::saturating_mul(
        detail::saturating_mul(n_regions, samples_per_region),
        sampler_bytes_per_sample);
    total = detail::saturating_add(
        total, detail::saturating_mul(n_regions, bound_bytes_per_region));
    if (enumerable) {
      total = detail::saturating_add(
          total, detail::saturating_mul(n_regions, exact_bytes_per_region));
    }
    return detail::saturating_add(total, fixed_bytes);
  }
};

}  // namespace cuminlp::backend
