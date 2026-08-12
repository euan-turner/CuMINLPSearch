#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include "cuminlp/aggregate/bound.hpp"
#include "cuminlp/aggregate/cost.hpp"
#include "cuminlp/backend/aggregate/replay.cuh"
#include "cuminlp/backend/backend.hpp"
#include "cuminlp/backend/graph/replay.cuh"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"

// The aggregate backend's two roles, built together
// (design/AGGREGATE_BOUNDING.md §9).
namespace cuminlp::backend::aggregate
{

using cuminlp::aggregate::AggregateSampler;

/**
 * @brief The incumbent source: the existing point graph, stratified.
 *
 * A thin adapter, not a new graph. `graph::PointGraphReplay` already draws
 * `sample_points` points inside each of its subregions, checks every one
 * against every constraint with the exact point predicate, masks the
 * infeasible ones to `+inf` and returns the best with its witness -- so its
 * candidate is an objective value attained at a feasible point by
 * construction (§6.2). Nothing about that needed changing; what changed is
 * only which partition it is handed and how big a budget it gets.
 *
 * Two differences from the bounder, both deliberate:
 *
 * - **A coarser partition, separately budgeted** (§6.3). This is a
 *   stratification device, not a bounding device.
 * - **An unsegmented epilogue.** The incumbent is global; which child the
 *   best sampled point fell in does not matter, so this is one ArgMin over
 *   everything rather than the bounder's `k` segments. Stratifying by the
 *   branch slots is what still guarantees each child its own share of draws.
 */
template<typename T>
class StratifiedSampler : public AggregateSampler<T>
{
public:
  static StratifiedSampler build(const model::Problem<T>& problem,
                                 const region::Composition& composition,
                                 std::size_t n_regions,
                                 std::size_t sample_points,
                                 std::size_t budget_bytes = 0,
                                 bool report_build = false)
  {
    return StratifiedSampler(graph::PointGraphReplay<T>::build(problem,
                                                               composition,
                                                               n_regions,
                                                               budget_bytes,
                                                               sample_points,
                                                               report_build));
  }

  backend::CandidateResult<T> sample(
      std::span<const cuminlp::aggregate::AggregateRegion<T>> regions,
      std::uint64_t salt) override
  {
    if (regions.size() != 1) {
      throw cuminlp::InvalidConfiguration(
          "StratifiedSampler: batched launches are not built yet "
          "(design/AGGREGATE_BOUNDING.md §14.1)");
    }
    // The sampler reads `partition.launch` and nothing else -- it has no use
    // for the branch/refine split beyond the stratification that split
    // already produced. The driver passes a partition built at the *sampler's*
    // budget, which is why this is a different (coarser) partition from the
    // one the bounder was handed for the same node.
    backend::Region<T> const r {regions[0].box, regions[0].partition.launch};
    return replay_.sample(r, salt);
  }

  std::size_t n_samples() const override { return replay_.n_samples(); }

  std::size_t n_regions() const { return replay_.n_regions(); }

private:
  explicit StratifiedSampler(graph::PointGraphReplay<T>&& replay)
      : replay_(std::move(replay))
  {
  }

  graph::PointGraphReplay<T> replay_;
};

/**
 * @brief Builds the aggregate backend's roles, and prices them beforehand.
 *
 * A separate hierarchy from `backend::RegionBackendFactory` on purpose (§1):
 * the two backends are compared, not merged, and a shared factory interface
 * would force the existing graph backend to answer for roles it does not
 * implement. There is correspondingly no abstract base here yet -- one
 * implementation, named concretely, until a second one earns the indirection.
 */
template<typename T>
class AggregateBackendFactory
{
public:
  /// `n_regions` is `N`; `branch_fan_out` is `k` and must divide it.
  std::unique_ptr<AggregateBounderReplay<T>> build_bounder(
      const model::Problem<T>& problem,
      const region::Composition& composition,
      std::size_t n_regions,
      std::size_t branch_fan_out,
      std::size_t budget_bytes,
      bool report_build = false) const
  {
    return std::make_unique<AggregateBounderReplay<T>>(
        AggregateBounderReplay<T>::build(problem,
                                         composition,
                                         n_regions,
                                         branch_fan_out,
                                         budget_bytes,
                                         report_build));
  }

  std::unique_ptr<StratifiedSampler<T>> build_sampler(
      const model::Problem<T>& problem,
      const region::Composition& composition,
      std::size_t n_regions,
      std::size_t sample_points,
      std::size_t budget_bytes,
      bool report_build = false) const
  {
    return std::make_unique<StratifiedSampler<T>>(
        StratifiedSampler<T>::build(problem,
                                    composition,
                                    n_regions,
                                    sample_points,
                                    budget_bytes,
                                    report_build));
  }

  /// Device bytes `build_bounder` would allocate for `n_regions`. The `k`
  /// outputs and the CUB scratch are excluded: both are `O(k)`/`O(sqrt(M))`
  /// against an `O(N)` total, so including them would complicate the fit for
  /// a term it cannot move.
  static std::size_t bounder_bytes(const model::Problem<T>& problem,
                                   std::size_t n_regions)
  {
    return detail::saturating_mul(n_regions,
                                  cuminlp::aggregate::bounder_element_bytes<T>(
                                      graph::buffer_node_count(problem)));
  }

  static std::size_t sampler_bytes(const model::Problem<T>& problem,
                                   std::size_t n_regions,
                                   std::size_t sample_points)
  {
    return detail::saturating_mul(
        detail::saturating_mul(n_regions, sample_points),
        cuminlp::aggregate::sampler_element_bytes<T>(
            graph::buffer_node_count(problem)));
  }
};

}  // namespace cuminlp::backend::aggregate
