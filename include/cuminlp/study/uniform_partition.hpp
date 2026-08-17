#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <string>

#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/composition.hpp"

// design/REFINEMENT_STUDY.md §7.2: a study-only CompositionPolicy matching
// Theorem 6.1's Definition 6.2 exactly -- every live dimension X_i is split
// into the same N_dim = 2^k parts, none left dead, so w(X_i,j) = w(X_i)/N_dim
// on every axis simultaneously and the theorem's N is the sweep's N_dim, not
// a shared budget spread unevenly the way BisectionBudgetCompositionPolicy
// spends B (design/BUDGETED_PARTITION.md §7). Used only by
// source/gams/refinement_study.cu; the solve path
// (aggregate_solve/gams_solve) is untouched and still uses
// BisectionBudgetCompositionPolicy.
namespace cuminlp::study
{

template<typename T>
class UniformCompositionPolicy : public policy::CompositionPolicy<T>
{
public:
  // `k` -- bisections applied to *every* live dimension. Must be >= 1 for
  // the same reason as BisectionBudgetCompositionPolicy's B: a slot with
  // k = 0 would not exist.
  explicit UniformCompositionPolicy(
      std::size_t k, config::SearchCalibration calibration = {})
      : policy::CompositionPolicy<T>(calibration)
      , k_(k)
  {
    if (k_ < 1) {
      throw InvalidConfiguration(
          "UniformCompositionPolicy: k must be at least 1 (b_j = 0 means "
          "the slot does not exist)");
    }
  }

  std::size_t k() const { return k_; }

  using policy::CompositionPolicy<T>::max_cycle_size;

  // Pure function of composition alone: every live slot this policy ever
  // emits has width 2^k_, so the total is determined by slot count -- N =
  // 2^(k_ * n_live), Definition 6.2's N_dim raised to the live dimension.
  std::size_t n_regions(const region::Composition& composition) const override
  {
    return std::size_t {1} << (k_ * composition.size());
  }

  region::SlotAssignment choose(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const override
  {
    assert(box.size() == var_kinds.size());

    std::size_t n_live = 0;
    for (std::size_t vid = 0; vid < box.size(); ++vid) {
      if (box[vid].lb < box[vid].ub) {
        if (var_kinds[vid] != model::VarKind::Continuous) {
          throw InvalidConfiguration(
              "UniformCompositionPolicy only supports Continuous variables "
              "(Definition 6.2's uniform subdivision has no enumerate-slot "
              "analogue); var " + std::to_string(vid) + " is not Continuous");
        }
        ++n_live;
      }
    }

    if (n_live == 0) {
      return region::SlotAssignment {};  // fully resolved box
    }
    if (n_live > max_cycle_size()) {
      throw InvalidConfiguration(
          "UniformCompositionPolicy: " + std::to_string(n_live)
          + " live dimensions exceeds max_cycle_size ("
          + std::to_string(max_cycle_size()) + ")");
    }
    // k_ * n_live > 62 would overflow the 2^... region count before it ever
    // reaches the device -- the same 62-bit guard
    // BisectionBudgetCompositionPolicy places on B alone, thrown as
    // ResourceExhausted (not InvalidConfiguration) because a smaller k or a
    // lower-dimensional box is a legitimate retry, exactly like a device
    // that is simply smaller.
    if (k_ > 62 / n_live) {
      throw ResourceExhausted(
          "UniformCompositionPolicy: k = " + std::to_string(k_) + " over "
          + std::to_string(n_live) + " live dimensions would need N = 2^"
          + std::to_string(k_ * n_live) + " regions");
    }

    region::SlotAssignment out {};
    std::size_t const w = std::size_t {1} << k_;
    for (std::size_t vid = 0; vid < box.size(); ++vid) {
      if (box[vid].lb < box[vid].ub) {
        out.composition.kinds.push_back(region::SlotKind::Continuous);
        out.var_ids.push_back(vid);
        out.widths.push_back(w);
        out.domain_sizes.push_back(w);  // exact split: never a duplicate
      }
    }
    return out;
  }

private:
  std::size_t k_;
};

}  // namespace cuminlp::study
