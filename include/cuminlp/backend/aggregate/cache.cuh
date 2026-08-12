#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "cuminlp/aggregate/driver.hpp"
#include "cuminlp/backend/aggregate/factory.cuh"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"

// The aggregate driver's role cache (design/AGGREGATE_BOUNDING.md §9).
namespace cuminlp::backend::aggregate
{

/**
 * @brief Holds one bounder and one sampler, rebuilding on a composition change.
 *
 * **Capacity one, not one per composition.** Each graph is sized to fill its
 * whole share of the device budget, so two of them do not fit -- a cache that
 * kept every composition it saw ran the device out of memory on the second
 * distinct one. That is the same pressure `search::BackendCache` exists to
 * manage for the other backend; here the budget arithmetic is simpler (`N` and
 * `k` are fixed solve-wide, so every graph of a given role costs the same) and
 * so is the answer: hold one, and rebuild when the composition changes.
 *
 * The launch composition changes when the refine's greedy allocation spreads
 * its budget over a different number of slots, which is a property of the
 * box rather than of the solve -- so a run whose live-variable count moves
 * around can rebuild repeatedly. `graphs_rebuilt()` counts exactly that, and
 * it is reported: a run that spends its time rebuilding graphs is a finding
 * for §14, not something to leave silently in the profile.
 */
template<typename T>
class GraphRoleCache : public cuminlp::aggregate::AggregateRoleCache<T>
{
public:
  GraphRoleCache(const model::Problem<T>& problem,
                 std::size_t total_fan_out,
                 std::size_t branch_fan_out,
                 std::size_t sample_fan_out,
                 std::size_t sample_points,
                 std::size_t bounder_budget,
                 std::size_t sampler_budget,
                 bool report_build)
      : problem_(problem)
      , n_(total_fan_out)
      , k_(branch_fan_out)
      , s_(sample_fan_out)
      , sample_points_(sample_points)
      , bounder_budget_(bounder_budget)
      , sampler_budget_(sampler_budget)
      , report_build_(report_build)
  {
  }

  cuminlp::aggregate::AggregateBounder<T>& bounder(
      const region::Composition& composition) override
  {
    if (bounder_ && bounder_composition_ == composition) {
      return *bounder_;
    }
    // Released before the replacement is built, not after: the two together
    // do not fit, which is the whole reason this cache holds one.
    if (bounder_) {
      bounder_.reset();
      ++graphs_rebuilt_;
    }
    bounder_ = factory_.build_bounder(
        problem_, composition, n_, k_, bounder_budget_, report_build_);
    bounder_composition_ = composition;
    ++graphs_built_;
    return *bounder_;
  }

  cuminlp::aggregate::AggregateSampler<T>& sampler(
      const region::Composition& composition) override
  {
    if (sampler_ && sampler_composition_ == composition) {
      return *sampler_;
    }
    if (sampler_) {
      sampler_.reset();
      ++graphs_rebuilt_;
    }
    sampler_ = factory_.build_sampler(problem_,
                                      composition,
                                      s_,
                                      sample_points_,
                                      sampler_budget_,
                                      report_build_);
    sampler_composition_ = composition;
    ++graphs_built_;
    return *sampler_;
  }

  std::size_t graphs_built() const override { return graphs_built_; }

  std::size_t graphs_rebuilt() const override { return graphs_rebuilt_; }

private:
  const model::Problem<T>& problem_;
  AggregateBackendFactory<T> factory_;
  std::unique_ptr<AggregateBounderReplay<T>> bounder_;
  std::unique_ptr<StratifiedSampler<T>> sampler_;
  region::Composition bounder_composition_ {};
  region::Composition sampler_composition_ {};
  std::size_t n_;
  std::size_t k_;
  std::size_t s_;
  std::size_t sample_points_;
  std::size_t bounder_budget_;
  std::size_t sampler_budget_;
  bool report_build_;
  std::size_t graphs_built_ = 0;
  std::size_t graphs_rebuilt_ = 0;
};

}  // namespace cuminlp::backend::aggregate
