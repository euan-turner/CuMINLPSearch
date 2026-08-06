#include <cstddef>

#include "cuminlp/config/run_spec.hpp"

#include <catch2/catch_test_macros.hpp>

#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/dag.hpp"

// nvs09's hand-built Problem, shared with policy_catalogue_test.cpp -- see
// that file's comment on why a hand-built Problem rather than a parsed
// .gms fixture.
#include "nvs09_problem.hpp"

using cuminlp::PolicyProfile;
using cuminlp::ProblemProfile;
using cuminlp::SearchCalibration;
using cuminlp::backend::RegionCostModel;
using cuminlp::config::OverrideSet;
using cuminlp::config::Provenance;
using cuminlp::config::resolve;
using cuminlp::config::RunSpec;
using cuminlp::config::to_string;

namespace
{

RegionCostModel graph_cost(const cuminlp::dag::Problem<double>& problem)
{
  return cuminlp::backend::graph::cost_model_for<double>(problem);
}

}  // namespace

// ---------------------------------------------------------------------------
// provenance
// ---------------------------------------------------------------------------

TEST_CASE("resolve reports the caller's base_source with no overrides given",
          "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  RunSpec const named = resolve(discrete, profile, calibration,
                                graph_cost(problem), Provenance::Named);
  CHECK(named.source == Provenance::Named);

  RunSpec const autos = resolve(discrete, profile, calibration,
                                graph_cost(problem), Provenance::Auto);
  CHECK(autos.source == Provenance::Auto);
}

TEST_CASE("any override forces Overridden regardless of base_source",
          "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  OverrideSet overrides;
  overrides.max_slots = 3;

  RunSpec const spec = resolve(discrete, profile, calibration,
                               graph_cost(problem), Provenance::Named,
                               overrides);
  CHECK(spec.source == Provenance::Overridden);
  CHECK(spec.max_slots == 3);
}

TEST_CASE("to_string spells every Provenance value", "[run_spec]")
{
  CHECK(to_string(Provenance::Auto) == "auto");
  CHECK(to_string(Provenance::Named) == "named");
  CHECK(to_string(Provenance::Overridden) == "overridden");
}

// ---------------------------------------------------------------------------
// override folding
// ---------------------------------------------------------------------------

TEST_CASE("an unset override leaves the resolved value untouched",
          "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  RunSpec const baseline = resolve(discrete, profile, calibration,
                                   graph_cost(problem), Provenance::Auto);

  OverrideSet overrides;
  overrides.sample_points = 99;
  RunSpec const spec = resolve(discrete, profile, calibration,
                               graph_cost(problem), Provenance::Auto,
                               overrides);

  CHECK(spec.sample_points == 99);
  CHECK(spec.fan_out.partition_num() == baseline.fan_out.partition_num());
  CHECK(spec.fan_out.enumerate_cap() == baseline.fan_out.enumerate_cap());
  CHECK(spec.max_slots == baseline.max_slots);
}

TEST_CASE(
    "overriding partition_num without enumerate_cap makes enumerate_cap "
    "follow it",
    "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  OverrideSet overrides;
  overrides.partition_num = 5;

  RunSpec const spec = resolve(discrete, profile, calibration,
                               graph_cost(problem), Provenance::Auto,
                               overrides);
  CHECK(spec.fan_out.partition_num() == 5);
  CHECK(spec.fan_out.enumerate_cap() == 5);
}

TEST_CASE(
    "overriding both partition_num and enumerate_cap keeps them "
    "independent",
    "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  OverrideSet overrides;
  overrides.partition_num = 5;
  overrides.enumerate_cap = 2;

  RunSpec const spec = resolve(discrete, profile, calibration,
                               graph_cost(problem), Provenance::Auto,
                               overrides);
  CHECK(spec.fan_out.partition_num() == 5);
  CHECK(spec.fan_out.enumerate_cap() == 2);
}

TEST_CASE("OverrideSet::any is false with nothing set", "[run_spec]")
{
  OverrideSet const overrides;
  CHECK_FALSE(overrides.any());
}
