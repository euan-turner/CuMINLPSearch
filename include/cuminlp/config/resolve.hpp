#pragma once

#include <algorithm>
#include <string>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/config/catalogue.hpp"
#include "cuminlp/config/footprint.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/format.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"

namespace cuminlp::config
{

/// The four resolved shape numbers for one run.
struct ResolvedShape
{
  std::size_t partition_num = 0;  ///< number of partitions to split
                                  // a variable domain into
  std::size_t enumerate_cap = 0;  ///< width of an integer domain at which to
                                  // enumerate it
  std::size_t sample_points =
      0;  ///< number of points to sample from a variable
          // subdomain in a single evaluation
  std::size_t max_cycle_size = 0;  ///< max number of variables to
                                   // partition/enumerate at once
};

/// Fraction of free device memory the resolver's fit will spend on the
/// widest composition (moved from source/gams/solve.cu, unchanged -- see
/// design/RUNTIME_SHAPE.md): the resolver is where budget-fitting happens
/// now, so this constant belongs beside it rather than in the CLI.
inline constexpr double auto_budget_fraction = 0.67;

namespace detail
{

/// The two-phase partition_num fit, shared by PartitionRule::FitToCoverage
/// (which widens after) and the single fit PartitionRule::Pin still needs for
/// max_cycle_size. `ec_follows_partition` is EnumerateRule::FollowPartition:
/// when set, enumerate_cap tracks whatever partition_num candidate is being
/// tried during the scan itself, not a value fixed before the loop starts.
inline std::size_t fit_at(const ProblemProfile& problem,
                          const backend::RegionCostModel& cost,
                          std::size_t partition_num,
                          std::size_t enumerate_cap,
                          bool ec_follows_partition,
                          std::size_t sample_points,
                          std::size_t budget,
                          std::size_t ceiling)
{
  region::FanOutSpec const fan_out {
      partition_num, ec_follows_partition ? partition_num : enumerate_cap};
  // The resolver is a pure function of counts, calibration and the backend's
  // cost coefficients -- not of any live Problem<T> or backend instance.
  return auto_max_cycle_size(problem.num_binary,
                             problem.num_integer,
                             problem.num_continuous,
                             cost,
                             fan_out,
                             sample_points,
                             budget,
                             ceiling);
}

}  // namespace detail

/**
 * @brief Turn a profile into a concrete search shape for this problem and
 *        device.
 *
 * Coverage first, width second: the variable count sets a target, and
 * partition_num is fitted to it in two phases when the profile asks for that
 * rather than looked up from a table. enumerate_cap is resolved from
 * the problem's integers alone, independent of partition_num, unless the
 * profile explicitly asks to keep that coupling (EnumerateRule::
 * FollowPartition). max_cycle_size is whatever the fit already bought, or a
 * profile-specific fallback when there is no device budget to fit against at
 * all.
 *
 * Every value this produces still passes through FanOutSpec's own
 * constructor, so a profile that pins a nonsensical number (partition_num
 * < 2, enumerate_cap < 1) throws InvalidConfiguration exactly as a
 * hand-typed flag would.
 *
 * Named `resolve_shape` rather than `resolve`: `config/run_spec.hpp`'s
 * `resolve()` is the one public entry point that folds overrides and
 * provenance on top of this (design/MODULE_REFACTOR.md §9 lists exactly one
 * `resolve` in config's public surface), and this is the shape-fitting core
 * it calls.
 */
inline ResolvedShape resolve_shape(const PolicyProfile& profile,
                                   const ProblemProfile& problem,
                                   const SearchCalibration& calibration,
                                   const backend::RegionCostModel& cost)
{
  // Bounds how many slots the fits below will ever try. Normally the
  // problem's own live-variable count (capped at kMaxSlots); a
  // CycleRule::Pin instead makes that bound the pin itself, so every branch
  // below -- including the have-budget ones, which otherwise never consult
  // profile.cycle at all -- naturally settles on exactly that many slots
  // rather than auto-fitting as many as the budget affords.
  std::size_t const target = profile.cycle.mode == CycleRule::Mode::Pin
      ? profile.cycle.pinned
      : std::min(problem.num_binary + problem.num_integer + problem.num_continuous,
                 kMaxSlots);

  bool const ec_follows_partition =
      profile.enumerate.mode == EnumerateRule::Mode::FollowPartition;

  // Resolved before partition_num whenever it doesn't depend on it;
  // FollowPartition is resolved against the final partition_num below
  // instead, including inside the fit loop itself (detail::fit_at).
  std::size_t enumerate_cap = 0;
  if (profile.enumerate.mode == EnumerateRule::Mode::CoverDomains) {
    // std::clamp requires low <= high, which a ceiling of 0 would violate;
    // two sequential clamps instead produce 0 in that case and let
    // FanOutSpec's ctor raise it as InvalidConfiguration, the same as
    // any other Pin below its floor.
    enumerate_cap =
        std::min(std::max(problem.largest_integer_domain, std::size_t {1}),
                 profile.enumerate.ceiling);
  } else if (profile.enumerate.mode == EnumerateRule::Mode::Pin) {
    enumerate_cap = profile.enumerate.pinned;
  }

  std::size_t partition_num = 0;
  std::size_t max_cycle_size = 0;

  bool const have_budget = calibration.free_device_bytes > 0;
  std::size_t const budget = have_budget
      ? static_cast<std::size_t>(
            static_cast<double>(calibration.free_device_bytes)
            * auto_budget_fraction)
      : 0;

  if (profile.partition.mode == PartitionRule::Mode::Pin) {
    partition_num = profile.partition.pinned;
    if (ec_follows_partition) {
      enumerate_cap = partition_num;
    }
    max_cycle_size = have_budget ? detail::fit_at(problem,
                                                  cost,
                                                  partition_num,
                                                  enumerate_cap,
                                                  false,
                                                  profile.sample_points,
                                                  budget,
                                                  target)
                                 : profile.cycle.pinned;
  } else if (!have_budget) {
    // No budget to fit against: scanning would charge every
    // candidate q the same 0-byte "footprint" (0 > budget(0) is false for
    // no q), so the naive loop below would never break and would report
    // partition_num == partition_ceiling for having "fit" nothing at all.
    // Fall back to the floor phase 1 itself starts from, and to the
    // profile's own pinned cycle size -- the same fallback CycleRule::Pin
    // gives a Pin partition above.
    partition_num = 2;
    if (ec_follows_partition) {
      enumerate_cap = partition_num;
    }
    max_cycle_size = profile.cycle.pinned;
  } else {
    // Two phases. Phase 1: the widest cap at the cheapest legal
    // fan-out. `achieved` is deliberately not updated inside phase 2's loop
    // -- it is the coverage phase 1 already bought, and phase 2 only asks
    // how wide partition_num can go while giving none of it back.
    std::size_t const achieved = detail::fit_at(problem,
                                                cost,
                                                2,
                                                enumerate_cap,
                                                ec_follows_partition,
                                                profile.sample_points,
                                                budget,
                                                target);
    partition_num = 2;
    for (std::size_t q = 3; q <= partition_ceiling; ++q) {
      std::size_t const fitted = detail::fit_at(problem,
                                                cost,
                                                q,
                                                enumerate_cap,
                                                ec_follows_partition,
                                                profile.sample_points,
                                                budget,
                                                target);
      if (fitted < achieved) {
        break;
      }
      partition_num = q;
    }
    if (ec_follows_partition) {
      enumerate_cap = partition_num;
    }
    max_cycle_size = achieved;
  }

  // A fitted 0 (not even one slot fits) becomes 1, so the run proceeds to
  // the build's own out-of-memory report rather than refusing to start.
  max_cycle_size = std::max(max_cycle_size, std::size_t {1});

  ResolvedShape shape;
  shape.partition_num = partition_num;
  shape.enumerate_cap = enumerate_cap;
  shape.sample_points = profile.sample_points;
  shape.max_cycle_size = max_cycle_size;
  // Constructs-and-discards purely to validate: a Pin resolved with no
  // budget to fit against never passes through FanOutSpec inside the fit
  // loop above, so this is what catches e.g. a Pin below FanOutSpec's own
  // floors, the same way a hand-typed --partition-num/--enumerate-cap would.
  region::FanOutSpec {shape.partition_num, shape.enumerate_cap};
  return shape;
}

/**
 * @brief Explain a build that did not fit: where the size came from, and
 *        which knob to turn.
 *
 * The advice half of what the backend used to compose itself
 * (design/MODULE_REFACTOR.md §5.6). The region count is a *product* over
 * slots, so an out-of-budget request is usually out by orders of magnitude
 * and the raw byte figure alone tells a caller nothing actionable. This shows
 * the multiplication that produced it, then does the arithmetic the caller
 * would otherwise have to: the widest slot cap that would actually fit.
 *
 * That last part is the useful half, and it is why this lives here rather
 * than in the backend. It is costed against the whole *problem*, not against
 * prefixes of the composition that happened to fail: on a problem with both
 * binaries and continuous variables the failing composition's cheap prefix
 * (binaries at fan-out 2) is not what a cap of that size buys further down
 * the tree, where the resolved binaries have handed their slots to continuous
 * variables at `partition_num` each. And it charges every role the solve
 * holds for a composition at once, not the one that overflowed: an exact
 * graph overflowing at one element per region must not recommend a cap at
 * which the sampler, at `sample_points` elements per region, overflows in
 * turn. Those are the two ways this advice was wrong before it was a
 * resolver's job (RUNTIME_SHAPE.md §6.3, §6.4).
 *
 * `solve_samples_per_region` is the solve-wide setting rather than what the
 * failing role itself drew: the size that failed is one fact, and what the
 * recommendation must be costed against is another.
 */
inline std::string explain_over_budget(const backend::OverBudget& over,
                                       const ProblemProfile& problem)
{
  std::string msg = over.role + " needs "
      + cuminlp::detail::format_bytes(over.needed_bytes)
      + " of device memory, but only "
      + cuminlp::detail::format_bytes(over.budget_bytes) + " is available.\n";

  // Per-kind slot tally: "10 x IntegerEnumerate (fan-out 7 each)". Only
  // meaningful when the policy that built this composition has one uniform
  // FanOutSpec (GreedyEnumerate/WidthFirst) -- BisectionBudget's widths are
  // per-slot and per-box, so `over.fan_out` is nullopt there and this whole
  // tally, along with the --max-cycle-size advice below, is skipped in
  // favour of a plainer report.
  msg += "  composition: " + std::to_string(over.composition.size())
      + " live slot(s)";
  if (over.fan_out) {
    for (int k = 0; k < 4; ++k) {
      auto const kind = static_cast<region::SlotKind>(k);
      std::size_t slots = 0;
      for (region::SlotKind s : over.composition) {
        if (s == kind) {
          ++slots;
        }
      }
      if (slots > 0) {
        msg += "\n    " + std::to_string(slots) + " x "
            + region::slot_kind_name(kind) + " (fan-out "
            + std::to_string(region::slot_fan_out(kind, *over.fan_out))
            + " each)";
      }
    }
  }

  msg += "\n  -> " + cuminlp::detail::format_count(over.n_regions) + " regions";
  if (over.elements_per_region > 1) {
    msg += " x " + std::to_string(over.elements_per_region) + " sample points";
  }
  msg += "\n  x " + cuminlp::detail::format_bytes(over.bytes_per_element)
      + " per element (" + over.element_breakdown + ")"
      + "\n  = " + cuminlp::detail::format_bytes(over.needed_bytes) + '\n';

  if (!over.fan_out) {
    msg += "\n  Lowering --bisection-budget shrinks every slot's fan-out "
           "together (N = 2^B), which reduces the region count.";
    return msg;
  }

  std::size_t const best_slots =
      auto_max_cycle_size(problem.num_binary,
                          problem.num_integer,
                          problem.num_continuous,
                          over.cost,
                          *over.fan_out,
                          over.solve_samples_per_region,
                          over.budget_bytes,
                          kMaxSlots);

  if (best_slots > 0) {
    std::size_t const best_bytes =
        worst_composition_footprint(best_slots,
                                    problem.num_binary,
                                    problem.num_integer,
                                    problem.num_continuous,
                                    over.cost,
                                    *over.fan_out,
                                    over.solve_samples_per_region);
    msg += "\n  Acting on " + std::to_string(best_slots)
        + " variable(s) at a time instead of "
        + std::to_string(over.composition.size())
        + " keeps every composition the search can reach -- point, interval"
          " and exact graphs together -- within "
        + cuminlp::detail::format_bytes(best_bytes)
        // Deliberately still --max-cycle-size, not --max-slots: this string
        // is unchanged from before stage 5's CLI rename (out of that
        // stage's scope, design/MODULE_REFACTOR.md §7.4) and stage 7 is
        // behaviour-preserving by construction, so it stays exactly as it
        // was rather than being fixed in passing here. Flagged as a
        // follow-up, not acted on speculatively.
        + ": try --max-cycle-size=" + std::to_string(best_slots) + '\n';
  } else {
    msg += "\n  Even a single slot does not fit, so the per-element cost is "
           "the problem rather than the slot count: this Problem needs "
        + cuminlp::detail::format_bytes(over.bytes_per_element) + " per region for its "
        + over.element_inventory + "\n";
  }
  msg +=
      "  Lowering --partition-num / --enumerate-cap shrinks each slot's "
      "fan-out, which reduces the product too.";
  return msg;
}

}  // namespace cuminlp::config
