// GPU tests for GraphReplay (graph_replay.cuh) -- TEST_EXTENSION.md §5.
// Requires an actual CUDA device.
#include <cmath>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/graph_replay.cuh"

using cuminlp::Composition;
using cuminlp::composition_fan_out;
using cuminlp::ShapeMismatch;
using cuminlp::SlotKind;
using cuminlp::dag::IntervalGraphReplay;
using cuminlp::dag::PointGraphReplay;
using cuminlp::dag::Problem;
using cuminlp::dag::VarKind;

namespace {

// x in [0, 10], objective x*x, constraint x <= 5 (feasible over part of the
// domain: [0,5] is feasible, (5,10] is not).
Problem<double> make_problem()
{
  Problem<double> p;
  auto x = p.var(0.0, 10.0);
  p.set_objective(x * x);
  p.add_constraint(x, cuminlp::dag::Cmp::LE, 5.0);
  return p;
}

// x in [0, 10], objective x*x, constraint x <= -100: infeasible everywhere.
Problem<double> make_infeasible_problem()
{
  Problem<double> p;
  auto x = p.var(0.0, 10.0);
  p.set_objective(x * x);
  p.add_constraint(x, cuminlp::dag::Cmp::LE, -100.0);
  return p;
}

}  // namespace

static_assert(!std::is_copy_constructible_v<PointGraphReplay<double, 1, 4, 8>>);
static_assert(!std::is_copy_assignable_v<PointGraphReplay<double, 1, 4, 8>>);
static_assert(std::is_move_constructible_v<PointGraphReplay<double, 1, 4, 8>>);

TEST_CASE("GraphReplay::n_regions matches composition_fan_out for its Composition", "[graph_replay][5e]")
{
  Problem<double> p = make_problem();
  Composition<1> comp = {SlotKind::Continuous};

  auto replay = IntervalGraphReplay<double, 1, 4>::build(p, comp);
  CHECK(replay.n_regions() == composition_fan_out<4, 4>(comp));
  CHECK(replay.n_regions() == 4);
}

TEST_CASE("set_domain rejects a domain whose size doesn't match the problem's variable count",
         "[graph_replay][5]")
{
  Problem<double> p = make_problem();
  Composition<1> comp = {SlotKind::Continuous};
  auto replay = IntervalGraphReplay<double, 1, 4>::build(p, comp);

  std::vector<cu::interval<double>> wrong_size_domain = {{0.0, 10.0}, {0.0, 1.0}};
  std::array<std::size_t, 1> var_ids = {0};

  REQUIRE_THROWS_AS(replay.set_domain(wrong_size_domain, var_ids), ShapeMismatch);
}

TEST_CASE("A moved-from GraphReplay is not launchable", "[graph_replay][5]")
{
  Problem<double> p = make_problem();
  Composition<1> comp = {SlotKind::Continuous};
  auto replay = IntervalGraphReplay<double, 1, 4>::build(p, comp);

  auto moved_to = std::move(replay);

  std::vector<cu::interval<double>> domain = {{0.0, 10.0}};
  std::array<std::size_t, 1> var_ids = {0};

  // The moved-to object is still perfectly usable.
  REQUIRE_NOTHROW(moved_to.set_domain(domain, var_ids));
  REQUIRE_NOTHROW(moved_to.launch(0));

  // The moved-from object must throw rather than dereference a null device
  // pointer.
  CHECK_THROWS_AS(replay.set_domain(domain, var_ids), cuminlp::error);
  CHECK_THROWS_AS(replay.launch(0), cuminlp::error);
}

TEST_CASE("has_candidate(), candidate_point()'s NaN-ness, and feasibility agree", "[graph_replay][5c]")
{
  Composition<1> comp = {SlotKind::Continuous};
  std::array<std::size_t, 1> var_ids = {0};

  SECTION("a domain with a feasible point has a non-NaN candidate")
  {
    Problem<double> p = make_problem();
    auto replay = PointGraphReplay<double, 1, 4, 16>::build(p, comp);

    std::vector<cu::interval<double>> domain = {{0.0, 5.0}};  // entirely feasible (x <= 5)
    replay.set_domain(domain, var_ids, /*salt=*/1);
    replay.launch(0);

    REQUIRE(replay.has_candidate());
    auto point = replay.candidate_point();
    REQUIRE(point.size() == 1);
    CHECK(!std::isnan(point[0]));
    CHECK(replay.candidate() < std::numeric_limits<double>::max());
  }

  SECTION("an entirely infeasible domain has no candidate, and its witness is all-NaN")
  {
    Problem<double> p = make_infeasible_problem();
    auto replay = PointGraphReplay<double, 1, 4, 16>::build(p, comp);

    std::vector<cu::interval<double>> domain = {{0.0, 10.0}};
    replay.set_domain(domain, var_ids, /*salt=*/1);
    replay.launch(0);

    CHECK_FALSE(replay.has_candidate());
    auto point = replay.candidate_point();
    REQUIRE(point.size() == 1);
    CHECK(std::isnan(point[0]));
  }
}

TEST_CASE("Replay determinism: identical (domain, var_ids, salt) gives bitwise-identical results",
         "[graph_replay][5d]")
{
  Problem<double> p = make_problem();
  Composition<1> comp = {SlotKind::Continuous};
  auto replay = PointGraphReplay<double, 1, 4, 32>::build(p, comp);

  std::vector<cu::interval<double>> domain = {{0.0, 5.0}};
  std::array<std::size_t, 1> var_ids = {0};

  replay.set_domain(domain, var_ids, /*salt=*/7);
  replay.launch(0);
  double const first_candidate = replay.candidate();
  auto const first_point_span = replay.candidate_point();
  std::vector<double> first_point(first_point_span.begin(), first_point_span.end());

  replay.set_domain(domain, var_ids, /*salt=*/7);
  replay.launch(0);
  double const second_candidate = replay.candidate();
  auto const second_point_span = replay.candidate_point();
  std::vector<double> second_point(second_point_span.begin(), second_point_span.end());

  CHECK(first_candidate == second_candidate);
  REQUIRE(first_point.size() == second_point.size());
  for (std::size_t i = 0; i < first_point.size(); ++i) {
    CHECK(first_point[i] == second_point[i]);
  }
}

TEST_CASE("Replay determinism: different salts draw different sample points", "[graph_replay][5d]")
{
  Problem<double> p = make_problem();
  Composition<1> comp = {SlotKind::Continuous};
  auto replay = PointGraphReplay<double, 1, 4, 32>::build(p, comp);

  std::vector<cu::interval<double>> domain = {{0.0, 5.0}};
  std::array<std::size_t, 1> var_ids = {0};

  replay.set_domain(domain, var_ids, /*salt=*/1);
  replay.launch(0);
  double const candidate_salt1 = replay.candidate();

  replay.set_domain(domain, var_ids, /*salt=*/2);
  replay.launch(0);
  double const candidate_salt2 = replay.candidate();

  // Weak, but would catch a sampler that ignores salt entirely: with 32
  // samples over a smooth objective, two different seeds landing on exactly
  // the same minimiser is implausible.
  CHECK(candidate_salt1 != candidate_salt2);
}

TEST_CASE("The interval graph's result is bitwise-identical regardless of salt", "[graph_replay][5d]")
{
  // The interval graph's root kernel takes no salt argument at all -- salt
  // only ever reaches set_domain() as a parameter of the shared API surface
  // (point and interval graphs), and must be silently ignored here.
  Problem<double> p = make_problem();
  Composition<1> comp = {SlotKind::Continuous};
  auto replay = IntervalGraphReplay<double, 1, 4>::build(p, comp);

  std::vector<cu::interval<double>> domain = {{0.0, 10.0}};
  std::array<std::size_t, 1> var_ids = {0};

  replay.set_domain(domain, var_ids, /*salt=*/1);
  replay.launch(0);
  std::vector<double> obj_lb_salt1(replay.obj_lb().begin(), replay.obj_lb().end());

  replay.set_domain(domain, var_ids, /*salt=*/2);
  replay.launch(0);
  std::vector<double> obj_lb_salt2(replay.obj_lb().begin(), replay.obj_lb().end());

  REQUIRE(obj_lb_salt1.size() == obj_lb_salt2.size());
  for (std::size_t i = 0; i < obj_lb_salt1.size(); ++i) {
    CHECK(obj_lb_salt1[i] == obj_lb_salt2[i]);
  }
}
