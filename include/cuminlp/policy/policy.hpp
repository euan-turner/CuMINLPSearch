#pragma once

#include <cstddef>
#include <span>
#include <string>

#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"

namespace cuminlp::policy
{

/// Which CompositionPolicy subclass a profile names.
enum class PolicyKind
{
  GreedyEnumerate, // prioritises enumerating discrete variables first
  WidthFirst,    // prioritises splitting the largest continuous domains first
  BisectionBudget, // per-slot widths under a fixed N = 2^B region budget
                   // (design/BUDGETED_PARTITION.md); opt-in only, see
                   // config::policy_roster and --policy=bisection-budget
  RelativeBisectionBudget, // BisectionBudget's greedy heap compares each
                   // candidate's live width against its own original
                   // domain width, not the raw absolute width; opt-in via
                   // --policy=relative-bisection-budget, for comparison
                   // against BisectionBudget on problems whose variables
                   // span different absolute scales
};

// Decides, for a search-tree node's current box, which variables to act on
// next and how many slots to use. Must be a pure function of the box, the
// problem's variable kinds, and the frozen SearchCalibration -- nothing that
// can change during a run (see config::SearchCalibration).
//
// `choose()` must return an assignment whose `var_ids` are pairwise distinct
// and each index a live (lb < ub) dimension, and whose `widths` (parallel to
// `composition.kinds`/`var_ids`, design/BUDGETED_PARTITION.md §4) are the
// fan-outs `search::Node::materialise` and the device's apply_slots_kernel
// decode `sidx` against -- reading both off the one returned SlotAssignment
// makes them impossible to disagree, the same reason a shared FanOutSpec
// used to live here. `GreedyEnumCompositionPolicy` satisfies the var_ids
// half of the contract by construction (fill_binary/fill_integer/
// fill_continuous each visit distinct vids and partition by VarKind), and
// SearchDriver asserts it in debug builds.
template<typename T>
class CompositionPolicy
{
public:
  explicit CompositionPolicy(config::SearchCalibration calibration = {})
      : calibration_(calibration)
  {
    // 0 means "unset": default to the full search cap.
    if (calibration_.max_cycle_size == 0) {
      calibration_.max_cycle_size = config::kMaxSlots;
    }
    if (calibration_.max_cycle_size > config::kMaxSlots) {
      throw InvalidConfiguration(
          "max_cycle_size (" + std::to_string(calibration_.max_cycle_size)
          + ") exceeds kMaxSlots (" + std::to_string(config::kMaxSlots) + ")");
    }
  }

  virtual ~CompositionPolicy() = default;

  const config::SearchCalibration& calibration() const { return calibration_; }

  // Slots this policy will ever fill.
  std::size_t max_cycle_size() const { return calibration_.max_cycle_size; }

  virtual region::SlotAssignment choose(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const = 0;

  // Total children a Composition with these kinds produces under this
  // policy -- i.e. the `n_regions` a GraphReplay built for it will launch.
  // A pure function of `composition` alone (never of the box a particular
  // SlotAssignment came from): BackendCache/GraphReplay::build need this to
  // size a graph before any box exists, and every SlotAssignment sharing a
  // Composition under a given policy must agree on it, or two boxes routed
  // through the same cached graph would disagree about how many regions it
  // launches (design/BUDGETED_PARTITION.md §3, the `Π w_j = N` invariant).
  virtual std::size_t n_regions(const region::Composition& composition) const = 0;

private:
  config::SearchCalibration calibration_;
};

// True iff every var_id in `assignment` is unique and indexes a live
// (lb < ub) dimension of `box` -- CompositionPolicy's contract (§4.5), a
// debug-only check (see SearchDriver::solve()). GreedyEnumCompositionPolicy
// satisfies it by construction; nothing asserted it before this.
template<typename T>
bool assignment_is_distinct_and_live(const region::SlotAssignment& assignment,
                                     std::span<const cu::interval<T>> box)
{
  for (std::size_t j = 0; j < assignment.var_ids.size(); ++j) {
    std::size_t const vid = assignment.var_ids[j];
    if (!(box[vid].lb < box[vid].ub)) {
      return false;
    }
    for (std::size_t k = j + 1; k < assignment.var_ids.size(); ++k) {
      if (assignment.var_ids[k] == vid) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace cuminlp::policy
