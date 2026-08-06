#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "cuminlp/backend/cost_model.hpp"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/config/catalogue.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/config/resolve.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/search/budget.hpp"

// The single owner of a run's hyperparameters (design/MODULE_REFACTOR.md
// §7.1), so the `PARAMS` line becomes a serialisation of one value instead
// of an assembly of locals. `resolve()` here wraps `resolve_shape()`
// (config/resolve.hpp) with the experimental-override folding and
// provenance bookkeeping that used to be hand-written in solve.cu's `main`
// (design/MODULE_REFACTOR.md §7.2).
//
// `RunSpec::calibration`, `budgets`, `frontier`, `iter_limit` and
// `tolerance` are not filled in by `resolve()` -- they are not search-shape
// decisions, so they stay solve.cu's to set (§7.3) before a RunSpec is
// handed to a driver. `resolve()` fills exactly the fields that come from
// fitting the shape against the problem, device and policy: `fan_out`,
// `max_slots`, `sample_points`, `policy_kind`/`policy_name`, `source` and
// `overrides`.
namespace cuminlp::config
{

/// Device/host memory ceilings for one run. 0 means "size against a live
/// measurement" (free device memory / MemAvailable) -- the existing meaning
/// of `SearchDriver`'s `budget_bytes`/`host_budget_bytes` constructor
/// arguments, carried over unchanged.
struct Budgets
{
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;
};

/// Where a RunSpec's policy came from -- printed verbatim as PARAMS'
/// `source=`.
enum class Provenance
{
  Auto,
  Named,
  Overridden
};

inline std::string_view to_string(Provenance source)
{
  switch (source) {
    case Provenance::Auto:
      return "auto";
    case Provenance::Named:
      return "named";
    case Provenance::Overridden:
      return "overridden";
  }
  return "?";
}

/// Which of the resolved shape's four numbers a caller pinned by hand, and
/// to what -- exactly the flags `--partition-num`/`--enumerate-cap`/
/// `--sample-points`/`--max-slots` (or its deprecated alias
/// `--max-cycle-size`) accept. A field is set iff the corresponding flag was
/// given; `any()` is what flips a RunSpec's provenance to `Overridden`
/// regardless of how the policy itself was chosen.
struct OverrideSet
{
  std::optional<std::size_t> partition_num;
  std::optional<std::size_t> enumerate_cap;
  std::optional<std::size_t> sample_points;
  std::optional<std::size_t> max_slots;

  bool any() const
  {
    return partition_num || enumerate_cap || sample_points || max_slots;
  }
};

struct RunSpec
{
  // --- what the search does ---
  policy::PolicyKind policy_kind = policy::PolicyKind::GreedyByKind;
  region::FanOutSpec fan_out;
  std::size_t max_slots = 1;
  std::size_t sample_points = 1;

  // --- what it may spend ---
  Budgets budgets;
  search::FrontierPolicy frontier = search::FrontierPolicy::StopAtBudget;
  std::uint32_t iter_limit = 1000000;
  double tolerance = 1e-6;

  // --- what it was derived from ---
  SearchCalibration calibration;
  std::string_view policy_name;
  Provenance source = Provenance::Auto;
  OverrideSet overrides;
};

/**
 * @brief Fit a search shape to this problem/device/policy, then fold in any
 *        experimental overrides.
 *
 * `base_source` is `Named` or `Auto` depending on whether the caller already
 * looked the policy up by name or let `select_policy` choose it -- that
 * distinction is made before this is called and isn't recoverable from
 * `overrides` alone, so it's threaded through rather than reconstructed.
 * `Overridden` always wins over it, exactly as solve.cu's own
 * `any_override` check did before this existed.
 *
 * `calibration` here is the *selection* calibration (device memory only,
 * same as solve.cu's `selection_calibration` before this change) -- the
 * fuller calibration a policy is actually constructed against (also probing
 * multiprocessor count, and carrying the chosen `max_slots` as its
 * `max_cycle_size`) is still solve.cu's to build afterwards, same as today.
 */
inline RunSpec resolve(const PolicyProfile& policy,
                       const ProblemProfile& problem,
                       const SearchCalibration& calibration,
                       const backend::RegionCostModel& cost,
                       Provenance base_source,
                       const OverrideSet& overrides = {})
{
  ResolvedShape const shape = resolve_shape(policy, problem, calibration, cost);

  std::size_t const chosen_partition_num =
      overrides.partition_num.value_or(shape.partition_num);
  // Given --partition-num without --enumerate-cap, enumerate_cap follows it
  // (the old single-arg-FanOutSpec shorthand) rather than staying at
  // whatever the policy resolved -- unchanged from solve.cu's prior logic.
  std::size_t const chosen_enumerate_cap = overrides.enumerate_cap
      ? *overrides.enumerate_cap
      : (overrides.partition_num ? chosen_partition_num : shape.enumerate_cap);

  // Designated-initialised, not default-constructed-then-assigned:
  // FanOutSpec has no default constructor (partition_num < 2 must always be
  // rejected), so RunSpec has none either. `calibration` here is only the
  // selection calibration passed in above; solve.cu overwrites it with the
  // fuller one it builds the policy against (see solve.cu's `solve`).
  return RunSpec {
      .policy_kind = policy.kind,
      .fan_out =
          region::FanOutSpec {chosen_partition_num, chosen_enumerate_cap},
      .max_slots = overrides.max_slots.value_or(shape.max_cycle_size),
      .sample_points = overrides.sample_points.value_or(shape.sample_points),
      .budgets = {},
      .frontier = search::FrontierPolicy::StopAtBudget,
      .iter_limit = 1000000,
      .tolerance = 1e-6,
      .calibration = calibration,
      .policy_name = policy.name,
      .source = overrides.any() ? Provenance::Overridden : base_source,
      .overrides = overrides,
  };
}

}  // namespace cuminlp::config
