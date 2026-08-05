#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/search_sizing.hpp"
 

namespace cuminlp
{

/// Which CompositionPolicy subclass a profile names.
enum class PolicyKind
{
  GreedyByKind
};

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
  PolicyKind kind;
  PartitionRule partition;
  EnumerateRule enumerate;
  std::size_t sample_points;  ///< still a tabulated constant
  CycleRule cycle;
  std::string_view evidence;  ///< what backs this row, for --list-policies
  bool provisional;  ///< no measurement behind it yet
};

/// The problem-side facts the resolver and classifier need, and nothing
/// else: a pure function of this struct plus a SearchCalibration must
/// reproduce a resolved shape.
struct ProblemProfile
{
  std::size_t num_binary = 0;
  std::size_t num_integer = 0;
  std::size_t num_continuous = 0;
  std::size_t largest_integer_domain = 0;  ///< over the root box, 0 if none
  std::size_t buffer_nodes = 0;  ///< dag::buffer_node_count(problem)

  /// True iff the objective variable survived lowering as a real continuous
  /// variable (neither the `=E=` nor the inequality elimination pass could
  /// solve for it).
  bool objvar_kept = false;
};

/**
 * @brief Derive a ProblemProfile from a parsed problem's variable kinds and
 *        bounds.
 *
 * Does not (and cannot) set `objvar_kept`: that fact comes from how the GAMS
 * frontend lowered the objective, not from anything a bare `dag::Problem<T>`
 * records. Callers with a `gams::ParsedModel<T>` set it separately.
 */
template<typename T>
ProblemProfile profile_problem(const dag::Problem<T>& problem)
{
  ProblemProfile out;
  for (std::size_t i = 0; i < problem.var_kinds.size(); ++i) {
    switch (problem.var_kinds[i]) {
      case dag::VarKind::Binary:
        ++out.num_binary;
        break;
      case dag::VarKind::Integer: {
        ++out.num_integer;
        // Number of integers in [ceil(lb), floor(ub)] -- same snapping as
        // GreedyCompositionPolicy::integer_domain_size, and 0 (not an
        // underflowed huge size_t) when the box has no integer point at all.
        T const lo = std::ceil(problem.box_bounds[i].lb);
        T const hi = std::floor(problem.box_bounds[i].ub);
        std::size_t const domain =
            hi < lo ? 0 : static_cast<std::size_t>(hi - lo) + 1;
        out.largest_integer_domain = std::max(out.largest_integer_domain, domain);
        break;
      }
      case dag::VarKind::Continuous:
        ++out.num_continuous;
        break;
    }
  }
  out.buffer_nodes = dag::buffer_node_count(problem);
  return out;
}

/// The four resolved shape numbers for one run.
struct ResolvedShape
{
  std::size_t partition_num = 0; ///< number of partitions to split
                                 // a variable domain into
  std::size_t enumerate_cap = 0; ///< width of an integer domain at which to
                                 // enumerate it
  std::size_t sample_points = 0; ///< number of points to sample from a variable
                                 // subdomain in a single evaluation
  std::size_t max_cycle_size = 0; ///< max number of variables to
                                  // partition/enumerate at once
};

/// Fraction of free device memory the resolver's fit will spend on the
/// widest composition (moved from source/gams/solve.cu, unchanged -- see
/// design/RUNTIME_SHAPE.md): the resolver is where budget-fitting happens
/// now, so this constant belongs beside it rather than in the CLI.
inline constexpr double auto_budget_fraction = 0.67;

/// Ceiling on partition_num's own fit: a guard against absurdity, not
/// a measured limit -- source/power_series.cu bisects one variable 10,000
/// ways, so the machinery tolerates far more than this. Numerically equal to
/// kMaxSlots (composition_policy.hpp) today, but a distinct constant: one
/// bounds a slot's fan-out, the other bounds how many slots exist.
inline constexpr std::size_t partition_ceiling = 64;

namespace detail
{

/// The two-phase partition_num fit, shared by PartitionRule::FitToCoverage
/// (which widens after) and the single fit PartitionRule::Pin still needs for
/// max_cycle_size. `ec_follows_partition` is EnumerateRule::FollowPartition:
/// when set, enumerate_cap tracks whatever partition_num candidate is being
/// tried during the scan itself, not a value fixed before the loop starts.
inline std::size_t fit_at(const ProblemProfile& problem,
                          std::size_t partition_num,
                          std::size_t enumerate_cap,
                          bool ec_follows_partition,
                          std::size_t sample_points,
                          std::size_t budget,
                          std::size_t ceiling)
{
  FanOutSpec const fan_out {partition_num,
                            ec_follows_partition ? partition_num
                                                 : enumerate_cap};
  // The resolver is a pure function of counts and calibration, not of
  // any live Problem<T> -- and gams_solve's own pipeline only ever solves in
  // double, so double is what the footprint arithmetic is sized against
  // here, the same as everywhere else in that pipeline.
  return dag::auto_max_cycle_size<double>(problem.num_binary,
                                          problem.num_integer,
                                          problem.num_continuous,
                                          problem.buffer_nodes,
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
 */
inline ResolvedShape resolve(const PolicyProfile& profile,
                             const ProblemProfile& problem,
                             const SearchCalibration& calibration)
{
  std::size_t const target =
      std::min(problem.num_binary + problem.num_integer + problem.num_continuous,
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
    enumerate_cap = std::min(
        std::max(problem.largest_integer_domain, std::size_t {1}),
        profile.enumerate.ceiling);
  } else if (profile.enumerate.mode == EnumerateRule::Mode::Pin) {
    enumerate_cap = profile.enumerate.pinned;
  }

  std::size_t partition_num = 0;
  std::size_t max_cycle_size = 0;

  bool const have_budget = calibration.free_device_bytes > 0;
  std::size_t const budget = have_budget
      ? static_cast<std::size_t>(static_cast<double>(calibration.free_device_bytes)
                                 * auto_budget_fraction)
      : 0;

  if (profile.partition.mode == PartitionRule::Mode::Pin) {
    partition_num = profile.partition.pinned;
    if (ec_follows_partition) {
      enumerate_cap = partition_num;
    }
    max_cycle_size = have_budget
        ? detail::fit_at(problem, partition_num, enumerate_cap, false,
                         profile.sample_points, budget, target)
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
    std::size_t const achieved = detail::fit_at(
        problem, 2, enumerate_cap, ec_follows_partition, profile.sample_points,
        budget, target);
    partition_num = 2;
    for (std::size_t q = 3; q <= partition_ceiling; ++q) {
      std::size_t const fitted = detail::fit_at(
          problem, q, enumerate_cap, ec_follows_partition,
          profile.sample_points, budget, target);
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
  FanOutSpec {shape.partition_num, shape.enumerate_cap};
  return shape;
}

/// The roster: concrete, inspectable instances of the rule automatic
/// selection will eventually be. Order matches the table in
/// design/POLICY_SELECTION.md and is what select_policy's classification
/// indexes into by position.
inline constexpr std::array<PolicyProfile, 5> policy_roster = {{
    {"all-binary",
     PolicyKind::GreedyByKind,
     {PartitionRule::Mode::Pin, 2},
     {EnumerateRule::Mode::Pin, 0, 2},
     1,
     {CycleRule::Mode::FitDevice, 20},
     "autocorr_bern20_03.cu",
     false},
    {"discrete",
     PolicyKind::GreedyByKind,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::CoverDomains, 16, 0},
     5,
     {CycleRule::Mode::FitDevice, 7},
     "nvs09.cu, RUNTIME_SHAPE.md",
     false},
    {"mixed-binary",
     PolicyKind::GreedyByKind,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::FollowPartition, 0, 0},
     5,
     {CycleRule::Mode::FitDevice, 4},
     "no size split has been measured",
     true},
    {"mixed-all-small",
     PolicyKind::GreedyByKind,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::CoverDomains, 16, 0},
     10,
     {CycleRule::Mode::FitDevice, 4},
     "ex8_6_2 (continuous-only)",
     false},
    {"mixed-all-large",
     PolicyKind::GreedyByKind,
     {PartitionRule::Mode::FitToCoverage, 0},
     {EnumerateRule::Mode::CoverDomains, 16, 0},
     3,
     {CycleRule::Mode::FitDevice, 4},
     "sample_points=3 is a guess, not a measurement",
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
    const ProblemProfile& problem, [[maybe_unused]] const SearchCalibration& calibration)
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
  return num_live <= kMaxSlots ? policy_roster[3]   // mixed-all-small
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
inline bool is_applicable(const PolicyProfile& policy, const ProblemProfile& problem)
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

/**
 * @brief Construct the CompositionPolicy subclass a PolicyKind names.
 *
 * Today GreedyByKind is the only value, so this is a one-case switch -- but
 * the switch, not an if, is deliberate: adding a second PolicyKind without a
 * matching case here is a compiler warning away from being caught.
 */
template<typename T>
std::unique_ptr<CompositionPolicy<T>> make_policy(
    PolicyKind kind, FanOutSpec fan_out, SearchCalibration calibration)
{
  switch (kind) {
    case PolicyKind::GreedyByKind:
      return std::make_unique<GreedyCompositionPolicy<T>>(fan_out, calibration);
  }
  throw InvalidConfiguration("unknown PolicyKind");
}

}  // namespace cuminlp
