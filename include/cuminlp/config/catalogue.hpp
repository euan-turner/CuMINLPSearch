#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/policy/policy.hpp"

namespace cuminlp::config
{

/// How `partition_num` is chosen. Pin fixes it regardless of the device;
/// FitToCoverage runs the two-phase fit in `resolve`.
struct PartitionRule
{
  enum class Mode
  {
    FitToCoverage,
    Pin
  };
  Mode mode;
  std::size_t pinned = 0;  ///< Pin only
};

/// How `enumerate_cap` is chosen. CoverDomains makes it a property of
/// the problem's integers rather than of partition_num, decoupling the two;
/// FollowPartition keeps the old coupling for the one row that still wants
/// it; Pin fixes it outright.
struct EnumerateRule
{
  enum class Mode
  {
    CoverDomains,
    FollowPartition,
    Pin
  };
  Mode mode;
  std::size_t ceiling = 0;  ///< CoverDomains only: cap on the domain covered
  std::size_t pinned = 0;  ///< Pin only
};

/// How `max_cycle_size` is chosen. FitDevice fits it to the budget,
/// falling back to `pinned` when there is no budget to fit against (no live
/// GPU, or cudaMemGetInfo failed).
struct CycleRule
{
  enum class Mode
  {
    FitDevice,
    Pin
  };
  Mode mode;
  std::size_t pinned = 0;
};

/// A named bundle of rules, not a tuple of numbers: what `resolve`
/// evaluates against a ProblemProfile and a SearchCalibration to produce a
/// ResolvedShape.
struct PolicyProfile
{
  std::string_view name;
  policy::PolicyKind kind;
  PartitionRule partition;
  EnumerateRule enumerate;
  std::size_t sample_points;  ///< still a tabulated constant
  CycleRule cycle;
  std::string_view evidence;  ///< what backs this row, for --list-policies
  bool provisional;  ///< no measurement behind it yet
};

/// Ceiling on partition_num's own fit: a guard against absurdity, not a
/// measured limit -- source/power_series.cu bisects one variable 10,000
/// ways, so the machinery tolerates far more than this. Used to sit at
/// kMaxSlots (64), but that was low enough to actually bind: a live GPU's
/// true single-slot budget fit landed well past a million on every model
/// probed for width-first (circle.gms 889167, alkyl.gms 1039955, ex14_1_1.gms
/// 1352789), and mixed-all-small was silently capped at 64 rather than its
/// true fit (96/110) on two of those same models. Raised past every observed
/// true limit; the scan itself stays cheap regardless (multi-slot rows still
/// break out within a couple hundred iterations, their product blowing the
/// budget long before the ceiling matters).
inline constexpr std::size_t partition_ceiling = 2000000;

/// The roster: concrete, inspectable instances of the rule automatic
/// selection will eventually be. Order matches the table in
/// design/POLICY_SELECTION.md and is what select_policy's classification
/// indexes into by position.
inline constexpr std::array<PolicyProfile, 6> policy_roster = {{
    {"all-binary",
     policy::PolicyKind::GreedyEnumerate,
     {PartitionRule::Mode::Pin, 2},
     {EnumerateRule::Mode::Pin, 0, 2},
     1,
     {CycleRule::Mode::FitDevice, 20},
     "autocorr_bern20_03.cu",
     false},
    {"discrete",
     policy::PolicyKind::GreedyEnumerate,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::CoverDomains, 16, 0},
     5,
     {CycleRule::Mode::FitDevice, 7},
     "nvs09.cu, RUNTIME_SHAPE.md",
     false},
    {"mixed-binary",
     policy::PolicyKind::GreedyEnumerate,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::FollowPartition, 0, 0},
     5,
     {CycleRule::Mode::FitDevice, 4},
     "no size split has been measured",
     true},
    {"mixed-all-small",
     policy::PolicyKind::GreedyEnumerate,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::CoverDomains, 16, 0},
     10,
     {CycleRule::Mode::FitDevice, 4},
     "ex8_6_2 (continuous-only)",
     false},
    {"mixed-all-large",
     policy::PolicyKind::GreedyEnumerate,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::CoverDomains, 16, 0},
     3,
     {CycleRule::Mode::FitDevice, 4},
     "sample_points=3 is a guess, not a measurement",
     true},
    // Named-only: select_policy never returns this row (it classifies by
    // position into policy_roster[0..4], design/POLICY_SELECTION.md), so
    // WidthFirstCompositionPolicy is reachable only via --policy=width-first,
    // never automatically. cycle pins to 1: the policy's whole point is
    // spending every cycle on the single widest live variable, not splitting
    // fan-out across several at once -- CycleRule::Pin actually enforces that
    // now (config/resolve.hpp's `target`), where it used to be a no-op
    // whenever a device budget was available. FitToCoverage then finds the
    // widest partition_num (up to partition_ceiling) that single slot can
    // afford, rather than a hand-picked constant.
    {"width-first",
     policy::PolicyKind::WidthFirst,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::CoverDomains, 16, 0},
     10,
     {CycleRule::Mode::Pin, 1},
     "no size split has been measured",
     true},
}};

/// Roster lookup by name, for --policy=<name> and --list-policies. Returns
/// nullopt on an unknown name; the caller (main) is what turns that into
/// exit 2 with the roster listed.
inline std::optional<PolicyProfile> lookup_policy(std::string_view name)
{
  for (const PolicyProfile& p : policy_roster) {
    if (p.name == name) {
      return p;
    }
  }
  return std::nullopt;
}

/**
 * @brief The continuous count classification actually reasons over:
 *        `num_continuous`, discounted by one when the objective variable was
 *        kept only because neither elimination pass could solve for it.
 *
 * Shared by select_policy and is_applicable, since both classify by variable
 * kind and both must agree on what "no continuous variable" means. Not used
 * by resolve() -- the coverage target there still counts every slot the
 * kept objvar will actually occupy: the variable is discounted from
 * *classification*, never from *sizing*.
 */
inline std::size_t classified_continuous(const ProblemProfile& problem)
{
  return problem.num_continuous
      - (problem.objvar_kept && problem.num_continuous > 0 ? 1 : 0);
}

/**
 * @brief Pick a roster row from the problem's characteristics alone, with no
 *        --policy given.
 *
 * Evaluated in the table's order, which is what makes it total: rules 4/5
 * (the mixed-all size split) are the fall-through for anything rules 1-3
 * didn't already match, including a continuous-only NLP, which matches
 * neither an all-discrete rule nor rule 3.
 *
 * `calibration` is accepted (not just ProblemProfile) because the selection
 * rule sees the frozen calibration among its inputs; no predicate below
 * currently reads it; a future hardware-dependent split would add one
 * without changing this function's signature or its callers.
 *
 * The objective-variable discount: a kept objvar occupies a slot but
 * is not a degree of freedom, so it must not be what tips an otherwise
 * all-binary or discrete problem into a mixed row. Discounting it from the
 * continuous count used *here* (and only here -- resolve()'s coverage target
 * still counts every slot the variable will actually occupy) is the
 * minimal change consistent with that: a problem with exactly one
 * continuous variable, which is a kept objvar, classifies as if it had none.
 */
inline const PolicyProfile& select_policy(
    const ProblemProfile& problem,
    [[maybe_unused]] const SearchCalibration& calibration)
{
  std::size_t const continuous = classified_continuous(problem);
  std::size_t const num_live =
      problem.num_binary + problem.num_integer + problem.num_continuous;

  if (continuous == 0 && problem.num_integer == 0) {
    return policy_roster[0];  // all-binary
  }
  if (continuous == 0) {
    return policy_roster[1];  // discrete
  }
  if (problem.num_binary > 0 && problem.num_integer == 0) {
    return policy_roster[2];  // mixed-binary
  }
  return num_live <= kMaxSlots ? policy_roster[3]  // mixed-all-small
                               : policy_roster[4];  // mixed-all-large
}

/**
 * @brief Whether a *named* policy's rules make sense for this problem's
 *        variable kinds -- the check a `--policy=<name>` override needs that
 *        automatic selection doesn't, since select_policy only ever returns
 *        a row it already knows fits.
 *
 * Deliberately not the same predicate select_policy uses to choose *between*
 * applicable rows: the mixed-all size split (rule 4 vs 5) is a preference
 * between two rows that are both always mechanically sound for a continuous
 * problem, not a constraint, so overriding `--policy=mixed-all-small` on a
 * 100-variable model is accepted -- that A/B comparison is exactly what a
 * named override is for. What *is* rejected is a policy whose rules assume a
 * variable kind the problem doesn't have in a way that would misrepresent
 * the search: `all-binary`/`discrete` pin or cap fan-out at values tuned
 * for an all-discrete problem, and silently applying them to a continuous
 * dimension bisects it at width 2 -- not a crash, but not what the row's
 * name and evidence claim to do either.
 *
 * `mixed-all-small`/`mixed-all-large` are the universal fallback (their
 * rules are no-ops on any variable kind the problem lacks), so they are
 * always applicable.
 */
inline bool is_applicable(const PolicyProfile& policy,
                          const ProblemProfile& problem)
{
  std::size_t const continuous = classified_continuous(problem);
  if (policy.name == "all-binary") {
    return continuous == 0 && problem.num_integer == 0;
  }
  if (policy.name == "discrete") {
    return continuous == 0;
  }
  if (policy.name == "mixed-binary") {
    return problem.num_integer == 0;
  }
  return true;  // mixed-all-small, mixed-all-large: the universal fallback
}

}  // namespace cuminlp::config
