#include <cstddef>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/config/catalogue.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/config/resolve.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/gams.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/greedy_enum.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"

// nvs09's hand-built Problem, shared with source/nvs09.cu and
// test/source/gams_test.cpp: 10 integers on [3, 9], no continuous, no
// binaries. Used here (rather than a parsed .gms fixture) so the resolver
// and classifier tests need no GPU and no .gms file at all -- exactly the
// property design/POLICY_SELECTION.md asks of this header.
#include "nvs09_problem.hpp"

using cuminlp::backend::RegionCostModel;
using cuminlp::config::CycleRule;
using cuminlp::config::EnumerateRule;
using cuminlp::config::PartitionRule;
using cuminlp::config::PolicyProfile;
using cuminlp::config::ProblemProfile;
using cuminlp::config::ResolvedShape;
using cuminlp::config::SearchCalibration;
using cuminlp::model::VarKind;
using cuminlp::policy::PolicyKind;
using cuminlp::region::FanOutSpec;

namespace
{

// The CUDA-graph backend's cost coefficients, which is what the resolver used
// to reach for through ProblemProfile::buffer_nodes
// (design/MODULE_REFACTOR.md §5.5). Every fit below is priced against these.
template<typename T>
RegionCostModel graph_cost(const cuminlp::model::Problem<T>& problem)
{
  return cuminlp::backend::graph::cost_model_for<T>(problem);
}

}  // namespace

namespace
{

auto data_file(char const* name) -> std::string
{
  return std::string(CUMINLP_GAMS_TEST_DATA) + "/" + name;
}

}  // namespace

// ---------------------------------------------------------------------------
// roster lookup
// ---------------------------------------------------------------------------

TEST_CASE("lookup_policy finds every roster row by name", "[policy_catalogue]")
{
  for (char const* name : {"all-binary",
                           "discrete",
                           "mixed-binary",
                           "mixed-all-small",
                           "mixed-all-large"})
  {
    auto found = cuminlp::config::lookup_policy(name);
    REQUIRE(found.has_value());
    CHECK(found->name == std::string_view(name));
  }
}

TEST_CASE("lookup_policy returns nullopt for an unknown name",
          "[policy_catalogue]")
{
  CHECK_FALSE(cuminlp::config::lookup_policy("quadratic").has_value());
  CHECK_FALSE(cuminlp::config::lookup_policy("").has_value());
  CHECK_FALSE(
      cuminlp::config::lookup_policy("mixed").has_value());  // the old name
}

TEST_CASE("policy_roster rows carry the roster's stated constants",
          "[policy_catalogue]")
{
  auto const all_binary = *cuminlp::config::lookup_policy("all-binary");
  CHECK(all_binary.partition.mode == PartitionRule::Mode::Pin);
  CHECK(all_binary.partition.pinned == 2);
  CHECK(all_binary.enumerate.mode == EnumerateRule::Mode::Pin);
  CHECK(all_binary.enumerate.pinned == 2);
  CHECK(all_binary.sample_points == 1);
  CHECK(all_binary.cycle.pinned == 20);
  CHECK_FALSE(all_binary.provisional);

  auto const discrete = *cuminlp::config::lookup_policy("discrete");
  CHECK(discrete.partition.mode == PartitionRule::Mode::FitToCoverage);
  CHECK(discrete.enumerate.mode == EnumerateRule::Mode::CoverDomains);
  CHECK(discrete.enumerate.ceiling == 16);
  CHECK(discrete.cycle.pinned == 7);

  auto const mixed_binary = *cuminlp::config::lookup_policy("mixed-binary");
  CHECK(mixed_binary.enumerate.mode == EnumerateRule::Mode::FollowPartition);
  CHECK(mixed_binary.provisional);
}

// ---------------------------------------------------------------------------
// profile_problem and the resolver
// ---------------------------------------------------------------------------

TEST_CASE("profile_problem reads nvs09's ten [3, 9] integers",
          "[policy_catalogue]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile profile = cuminlp::config::profile_problem(problem);

  CHECK(profile.num_binary == 0);
  CHECK(profile.num_integer == 10);
  CHECK(profile.num_continuous == 0);
  CHECK(profile.largest_integer_domain == 7);  // [3, 9] has 7 integer points
  CHECK_FALSE(profile.objvar_kept);  // profile_problem never sets this
}

TEST_CASE("CoverDomains clamps enumerate_cap to the problem's largest domain",
          "[policy_catalogue]")
{
  // nvs09's domains (7) sit comfortably under discrete's ceiling (16), so the
  // clamp's upper bound is unexercised and the answer is the domain itself.
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");

  SearchCalibration calibration;
  calibration.free_device_bytes = 64ull * 1024 * 1024 * 1024;  // 64 GiB
  ResolvedShape const shape = cuminlp::config::resolve_shape(
      discrete, profile, calibration, graph_cost(problem));
  CHECK(shape.enumerate_cap == 7);
}

TEST_CASE(
    "resolve's two-phase fit re-derives nvs09's tuned shape: phase 2 "
    "widens partition_num after phase 1 has already given up coverage",
    "[policy_catalogue]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");

  // Every integer's domain is exactly 7, so for any q <= 7 the charge per
  // integer slot is max(q, enumerate_cap=7) == 7 -- identical to q == 2 -- and
  // only rises once q > 7. Picking the budget as exactly the footprint at cap
  // 7 (rather than a guessed round number) pins the boundary precisely,
  // regardless of nvs09's actual buffer-node count.
  RegionCostModel const cost = graph_cost(problem);
  auto const regions = [](std::size_t slots)
  {
    std::size_t r = 1;
    for (std::size_t i = 0; i < slots; ++i) {
      r *= 7;
    }
    return r;
  };
  std::size_t const footprint_at_7 = cost.bundle_bytes(
      regions(7), discrete.sample_points, /*enumerable=*/true);
  std::size_t const footprint_at_8 = cost.bundle_bytes(
      regions(8), discrete.sample_points, /*enumerable=*/true);
  REQUIRE(footprint_at_8 > footprint_at_7);  // sanity: charge does rise past 7

  SearchCalibration calibration;
  // budget = free_device_bytes * auto_budget_fraction (0.67); pad so
  // floating-point rounding can't drop budget just below footprint_at_7.
  calibration.free_device_bytes =
      static_cast<std::size_t>(static_cast<double>(footprint_at_7)
                                   / cuminlp::config::auto_budget_fraction
                               + 1024.0);

  ResolvedShape const shape =
      cuminlp::config::resolve_shape(discrete, profile, calibration, cost);

  CHECK(shape.max_cycle_size == 7);  // coverage sacrificed: 7 of 10
  CHECK(shape.partition_num == 7);  // phase 2 widened 2 -> 7 for free
  CHECK(shape.enumerate_cap == 7);
  CHECK(shape.max_cycle_size <= cuminlp::config::kMaxSlots);
}

TEST_CASE("resolve falls back to the profile's pinned cycle size with no "
          "device budget to fit against",
          "[policy_catalogue]")
{
  SearchCalibration const no_budget;  // free_device_bytes == 0 by default
  ProblemProfile problem;
  problem.num_integer = 10;
  problem.largest_integer_domain = 7;

  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");
  // No budget to fit against, so the cost model never gets consulted; an
  // all-zero one makes that explicit rather than incidental.
  RegionCostModel const unused_cost;
  ResolvedShape const shape =
      cuminlp::config::resolve_shape(discrete, problem, no_budget, unused_cost);
  CHECK(shape.partition_num == 2);  // the floor phase 1 itself starts from
  CHECK(shape.enumerate_cap == 7);  // CoverDomains doesn't need a budget
  CHECK(shape.max_cycle_size == 7);  // discrete's pinned fallback

  ProblemProfile all_binary_problem;
  all_binary_problem.num_binary = 20;
  PolicyProfile const all_binary =
      *cuminlp::config::lookup_policy("all-binary");
  ResolvedShape const pinned_shape = cuminlp::config::resolve_shape(
      all_binary, all_binary_problem, no_budget, unused_cost);
  CHECK(pinned_shape.partition_num == 2);
  CHECK(pinned_shape.enumerate_cap == 2);
  CHECK(pinned_shape.max_cycle_size == 20);  // all-binary's pinned fallback
}

TEST_CASE("a resolved shape's fields compose independently under override",
          "[policy_catalogue]")
{
  // The decoupling claim, exercised directly: overriding partition_num
  // (the way an experimental --partition-num flag would, on top of the
  // resolved shape) must not disturb the enumerate_cap CoverDomains already
  // resolved, and the overridden pair must still be a legal FanOutSpec.
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");

  SearchCalibration calibration;
  calibration.free_device_bytes = 64ull * 1024 * 1024 * 1024;
  ResolvedShape shape = cuminlp::config::resolve_shape(
      discrete, profile, calibration, graph_cost(problem));
  std::size_t const original_enumerate_cap = shape.enumerate_cap;

  shape.partition_num += 1;
  CHECK(shape.enumerate_cap == original_enumerate_cap);
  CHECK_NOTHROW((FanOutSpec {shape.partition_num, shape.enumerate_cap}));
}

// ---------------------------------------------------------------------------
// select_policy's classification, including the objvar discount
// ---------------------------------------------------------------------------

TEST_CASE("select_policy's five predicates at their boundaries",
          "[policy_catalogue]")
{
  SearchCalibration const calibration;

  SECTION("rule 1: no continuous, no integer -> all-binary")
  {
    ProblemProfile p;
    p.num_binary = 5;
    CHECK(cuminlp::config::select_policy(p, calibration).name == "all-binary");
  }

  SECTION("rule 2: no continuous, some integer -> discrete, binaries included")
  {
    ProblemProfile p;
    p.num_integer = 1;
    CHECK(cuminlp::config::select_policy(p, calibration).name == "discrete");

    ProblemProfile mixed_discrete;
    mixed_discrete.num_binary = 5;
    mixed_discrete.num_integer = 1;
    CHECK(cuminlp::config::select_policy(mixed_discrete, calibration).name
          == "discrete");
  }

  SECTION("rule 3: some continuous, some binary, no integer -> mixed-binary")
  {
    ProblemProfile p;
    p.num_continuous = 1;
    p.num_binary = 1;
    CHECK(cuminlp::config::select_policy(p, calibration).name
          == "mixed-binary");
  }

  SECTION("rule 4/5 split at num_live > kMaxSlots (64)")
  {
    ProblemProfile at_cap;
    at_cap.num_continuous = 64;
    CHECK(cuminlp::config::select_policy(at_cap, calibration).name
          == "mixed-all-small");

    ProblemProfile over_cap;
    over_cap.num_continuous = 65;
    CHECK(cuminlp::config::select_policy(over_cap, calibration).name
          == "mixed-all-large");
  }
}

TEST_CASE("a continuous-only problem falls through to mixed-all",
          "[policy_catalogue]")
{
  SearchCalibration const calibration;
  ProblemProfile p;
  p.num_continuous = 19;  // ex8_6_2's shape: no Binary/Integer section at all
  CHECK(cuminlp::config::select_policy(p, calibration).name
        == "mixed-all-small");
}

TEST_CASE("objvar_kept discounts exactly one continuous slot from "
          "classification, not from anything else",
          "[policy_catalogue]")
{
  SearchCalibration const calibration;

  ProblemProfile kept;
  kept.num_binary = 20;
  kept.num_continuous = 1;
  kept.objvar_kept = true;
  CHECK(cuminlp::config::select_policy(kept, calibration).name == "all-binary");

  // Same counts, but the objvar was NOT specially kept (an ordinary
  // continuous variable): the discount must not apply, and this must
  // classify as mixed rather than all-binary.
  ProblemProfile not_kept = kept;
  not_kept.objvar_kept = false;
  CHECK(cuminlp::config::select_policy(not_kept, calibration).name
        == "mixed-binary");

  // Two continuous variables, one of them a kept objvar: discounting one
  // still leaves a genuine continuous dimension, so this stays mixed.
  ProblemProfile two_continuous;
  two_continuous.num_binary = 20;
  two_continuous.num_continuous = 2;
  two_continuous.objvar_kept = true;
  CHECK(cuminlp::config::select_policy(two_continuous, calibration).name
        == "mixed-binary");
}

TEST_CASE("is_applicable rejects a named policy whose rules assume a "
          "variable kind the problem doesn't have",
          "[policy_catalogue]")
{
  PolicyProfile const all_binary =
      *cuminlp::config::lookup_policy("all-binary");
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");
  PolicyProfile const mixed_binary =
      *cuminlp::config::lookup_policy("mixed-binary");
  PolicyProfile const mixed_small =
      *cuminlp::config::lookup_policy("mixed-all-small");
  PolicyProfile const mixed_large =
      *cuminlp::config::lookup_policy("mixed-all-large");

  SECTION("all-binary: rejects any integer or continuous variable")
  {
    ProblemProfile p;
    p.num_binary = 5;
    CHECK(cuminlp::config::is_applicable(all_binary, p));

    ProblemProfile with_integer = p;
    with_integer.num_integer = 1;
    CHECK_FALSE(cuminlp::config::is_applicable(all_binary, with_integer));

    ProblemProfile with_continuous = p;
    with_continuous.num_continuous = 1;
    CHECK_FALSE(cuminlp::config::is_applicable(all_binary, with_continuous));
  }

  SECTION("discrete: rejects any continuous variable, integers/binaries fine")
  {
    ProblemProfile p;
    p.num_integer = 10;
    p.num_binary = 5;
    CHECK(cuminlp::config::is_applicable(discrete, p));

    ProblemProfile with_continuous = p;
    with_continuous.num_continuous = 1;
    CHECK_FALSE(cuminlp::config::is_applicable(discrete, with_continuous));
  }

  SECTION("mixed-binary: rejects any integer variable")
  {
    ProblemProfile p;
    p.num_continuous = 3;
    p.num_binary = 2;
    CHECK(cuminlp::config::is_applicable(mixed_binary, p));

    ProblemProfile with_integer = p;
    with_integer.num_integer = 1;
    CHECK_FALSE(cuminlp::config::is_applicable(mixed_binary, with_integer));
  }

  SECTION("mixed-all-small/large: the universal fallback, never rejected")
  {
    ProblemProfile anything;
    anything.num_binary = 20;
    anything.num_integer = 20;
    anything.num_continuous = 20;
    CHECK(cuminlp::config::is_applicable(mixed_small, anything));
    CHECK(cuminlp::config::is_applicable(mixed_large, anything));

    ProblemProfile empty;
    CHECK(cuminlp::config::is_applicable(mixed_small, empty));
    CHECK(cuminlp::config::is_applicable(mixed_large, empty));
  }

  SECTION("the objvar discount applies here too: a kept objvar doesn't "
          "block discrete/all-binary")
  {
    ProblemProfile p;
    p.num_integer = 10;
    p.num_continuous = 1;
    p.objvar_kept = true;
    CHECK(cuminlp::config::is_applicable(discrete, p));

    ProblemProfile not_kept = p;
    not_kept.objvar_kept = false;
    CHECK_FALSE(cuminlp::config::is_applicable(discrete, not_kept));
  }
}

// ---------------------------------------------------------------------------
// evidence: the three tuned instances still select their own row
// ---------------------------------------------------------------------------

TEST_CASE(
    "autocorr_bern20-03 (the all-binary row's evidence) selects " "all-binary",
    "[policy_catalogue]")
{
  auto parsed =
      cuminlp::gams::parse_file<double>(data_file("autocorr_bern20-03.gms"));

  ProblemProfile profile = cuminlp::config::profile_problem(parsed.problem);
  profile.objvar_kept = parsed.objvar_kept;

  // The inequality rewrite (frontend.cpp's eliminate_objective) substitutes
  // this file's objvar away entirely, so it never reaches ProblemProfile as
  // a continuous variable at all -- objvar_kept should already be false, not
  // merely discounted.
  CHECK_FALSE(parsed.objvar_kept);
  CHECK(profile.num_binary == 20);
  CHECK(profile.num_continuous == 0);

  SearchCalibration const calibration;
  CHECK(cuminlp::config::select_policy(profile, calibration).name
        == "all-binary");
}

TEST_CASE(
    "ex8_6_2 (the mixed-all-small row's evidence) selects " "mixed-all-small",
    "[policy_catalogue]")
{
  auto parsed = cuminlp::gams::parse_file<double>(data_file("ex8_6_2.gms"));

  ProblemProfile profile = cuminlp::config::profile_problem(parsed.problem);
  profile.objvar_kept = parsed.objvar_kept;

  CHECK(profile.num_binary == 0);
  CHECK(profile.num_integer == 0);
  CHECK(profile.num_continuous > 0);  // continuous-only

  SearchCalibration const calibration;
  CHECK(cuminlp::config::select_policy(profile, calibration).name
        == "mixed-all-small");
}

TEST_CASE("nvs09 (the discrete row's evidence) selects discrete",
          "[policy_catalogue]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);

  SearchCalibration const calibration;
  CHECK(cuminlp::config::select_policy(profile, calibration).name
        == "discrete");
}

// ---------------------------------------------------------------------------
// make_policy constructs the named CompositionPolicy subclass
// ---------------------------------------------------------------------------

TEST_CASE(
    "make_policy<GreedyEnumerate> constructs a working GreedyEnumCompositionPolicy",
    "[policy_catalogue]")
{
  auto policy = cuminlp::policy::make_policy<double>(
      PolicyKind::GreedyEnumerate, FanOutSpec {4}, 0, SearchCalibration {});
  REQUIRE(policy != nullptr);

  std::vector<cu::interval<double>> box = {{0.0, 1.0}};
  std::vector<VarKind> kinds = {VarKind::Binary};
  auto assignment = policy->choose(box, kinds);
  CHECK(assignment.composition[0]
        == cuminlp::region::SlotKind::BinaryEnumerate);
}
