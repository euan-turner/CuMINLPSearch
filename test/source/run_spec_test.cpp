#include <cstddef>

#include "cuminlp/config/run_spec.hpp"

#include <catch2/catch_test_macros.hpp>

#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/model/problem.hpp"

// nvs09's hand-built Problem, shared with policy_catalogue_test.cpp -- see
// that file's comment on why a hand-built Problem rather than a parsed
// .gms fixture.
#include "nvs09_problem.hpp"

using cuminlp::backend::RegionCostModel;
using cuminlp::config::OverrideSet;
using cuminlp::config::PolicyProfile;
using cuminlp::config::ProblemProfile;
using cuminlp::config::Provenance;
using cuminlp::config::resolve;
using cuminlp::config::RunSpec;
using cuminlp::config::SearchCalibration;
using cuminlp::config::to_string;

namespace
{

RegionCostModel graph_cost(const cuminlp::model::Problem<double>& problem)
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
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  RunSpec const named = resolve(
      discrete, profile, calibration, graph_cost(problem), Provenance::Named);
  CHECK(named.source == Provenance::Named);

  RunSpec const autos = resolve(
      discrete, profile, calibration, graph_cost(problem), Provenance::Auto);
  CHECK(autos.source == Provenance::Auto);
}

TEST_CASE("any override forces Overridden regardless of base_source",
          "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  OverrideSet overrides;
  overrides.max_slots = 3;

  RunSpec const spec = resolve(discrete,
                               profile,
                               calibration,
                               graph_cost(problem),
                               Provenance::Named,
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

TEST_CASE("an unset override leaves the resolved value untouched", "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  RunSpec const baseline = resolve(
      discrete, profile, calibration, graph_cost(problem), Provenance::Auto);

  OverrideSet overrides;
  overrides.sample_points = 99;
  RunSpec const spec = resolve(discrete,
                               profile,
                               calibration,
                               graph_cost(problem),
                               Provenance::Auto,
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
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  OverrideSet overrides;
  overrides.partition_num = 5;

  RunSpec const spec = resolve(discrete,
                               profile,
                               calibration,
                               graph_cost(problem),
                               Provenance::Auto,
                               overrides);
  CHECK(spec.fan_out.partition_num() == 5);
  CHECK(spec.fan_out.enumerate_cap() == 5);
}

TEST_CASE(
    "overriding both partition_num and enumerate_cap keeps them " "independent",
    "[run_spec]")
{
  auto problem = cuminlp::examples::nvs09::make_nvs09();
  ProblemProfile const profile = cuminlp::config::profile_problem(problem);
  PolicyProfile const discrete = *cuminlp::config::lookup_policy("discrete");
  SearchCalibration calibration;
  calibration.free_device_bytes = 1'000'000'000;

  OverrideSet overrides;
  overrides.partition_num = 5;
  overrides.enumerate_cap = 2;

  RunSpec const spec = resolve(discrete,
                               profile,
                               calibration,
                               graph_cost(problem),
                               Provenance::Auto,
                               overrides);
  CHECK(spec.fan_out.partition_num() == 5);
  CHECK(spec.fan_out.enumerate_cap() == 2);
}

TEST_CASE("OverrideSet::any is false with nothing set", "[run_spec]")
{
  OverrideSet const overrides;
  CHECK_FALSE(overrides.any());
}

// ---------------------------------------------------------------------------
// design/BUDGETED_PARTITION.md §3.2: bisection_budget's closed form
// ---------------------------------------------------------------------------

TEST_CASE("bisection_budget picks the largest power of two that fits",
          "[run_spec][3.2]")
{
  // A known triple, computed by hand: 50 buffers, 5 sample points.
  std::size_t const n_buffers = 50;
  std::size_t const sample_points = 5;
  RegionCostModel const cost =
      cuminlp::backend::graph::cost_model_for<double>(n_buffers);
  std::size_t const per_region = cost.bundle_bytes(1, sample_points, true);
  REQUIRE(per_region > 0);

  // Exactly 2^10 * per_region, so B must be exactly 10 -- the boundary case,
  // not just "close enough".
  std::size_t const budget = per_region * 1024;
  std::size_t const b = cuminlp::backend::graph::bisection_budget<double>(
      n_buffers, sample_points, budget);
  CHECK(b == 10);
}

TEST_CASE("bisection_budget's B is the tightest bracket the budget admits",
          "[run_spec][3.2]")
{
  std::size_t const n_buffers = 17;
  std::size_t const sample_points = 3;
  std::size_t const budget = 512u * 1024 * 1024;  // 512 MiB
  RegionCostModel const cost =
      cuminlp::backend::graph::cost_model_for<double>(n_buffers);

  std::size_t const b = cuminlp::backend::graph::bisection_budget<double>(
      n_buffers, sample_points, budget);

  // Property-based, not a hand-computed constant: 2^b regions fit, 2^(b+1)
  // do not -- the definition of "largest power of two that fits".
  CHECK(cost.bundle_bytes(std::size_t {1} << b, sample_points, true) <= budget);
  CHECK(cost.bundle_bytes(std::size_t {1} << (b + 1), sample_points, true)
        > budget);
}

TEST_CASE("bisection_budget saturates to 0 when even one region doesn't fit",
          "[run_spec][3.2]")
{
  std::size_t const n_buffers = 50;
  std::size_t const sample_points = 5;
  RegionCostModel const cost =
      cuminlp::backend::graph::cost_model_for<double>(n_buffers);
  std::size_t const per_region = cost.bundle_bytes(1, sample_points, true);
  REQUIRE(per_region > 1);

  CHECK(cuminlp::backend::graph::bisection_budget<double>(
            n_buffers, sample_points, per_region - 1)
        == 0);
  CHECK(cuminlp::backend::graph::bisection_budget<double>(
            n_buffers, sample_points, 0)
        == 0);
}

TEST_CASE("bisection_budget is monotone in the budget", "[run_spec][3.2]")
{
  std::size_t const n_buffers = 8;
  std::size_t const sample_points = 1;
  std::size_t prev = 0;
  for (std::size_t budget :
       {1ull, 1000ull, 1'000'000ull, 1'000'000'000ull, 1'000'000'000'000ull})
  {
    std::size_t const b = cuminlp::backend::graph::bisection_budget<double>(
        n_buffers, sample_points, budget);
    CHECK(b >= prev);
    prev = b;
  }
}
