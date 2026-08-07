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
};

// Decides, for a search-tree node's current box, which variables to act on
// next and how many slots to use. Must be a pure function of the box, the
// problem's variable kinds, and the frozen SearchCalibration -- nothing that
// can change during a run (see config::SearchCalibration).
//
// The policy also *owns* the FanOutSpec. This is deliberate: decoding a
// node's box (search::Node::materialise) needs both the
// SlotAssignment and the fan-outs its sidx was encoded against, and reading
// both off one object makes them impossible to disagree.
//
// `choose()` must return an assignment whose `var_ids` are pairwise distinct
// and each index a live (lb < ub) dimension
// -- GreedyEnumCompositionPolicy satisfies this by construction (fill_binary/
// fill_integer/fill_continuous each visit distinct vids and partition by
// VarKind), and SearchDriver asserts it in debug builds.
template<typename T>
class CompositionPolicy
{
public:
  explicit CompositionPolicy(region::FanOutSpec fan_out,
                             config::SearchCalibration calibration = {})
      : fan_out_(fan_out)
      , calibration_(calibration)
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

  // The fan-outs every consumer of this policy's SlotAssignments must decode
  // against. Not virtual: there is one answer per policy instance, fixed at
  // construction.
  const region::FanOutSpec& fan_out() const { return fan_out_; }

  const config::SearchCalibration& calibration() const { return calibration_; }

  // Slots this policy will ever fill.
  std::size_t max_cycle_size() const { return calibration_.max_cycle_size; }

  virtual region::SlotAssignment choose(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const = 0;

private:
  region::FanOutSpec fan_out_;
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
