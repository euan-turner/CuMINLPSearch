// GPU tests for GraphReplay (graph_replay.cuh) -- see TEST_EXTENSION.md.
// Requires an actual CUDA device.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/graph_replay.cuh"
#include "cuminlp/policy_catalogue.hpp"

using cuminlp::Composition;
using cuminlp::composition_fan_out;
using cuminlp::FanOutSpec;
using cuminlp::ShapeMismatch;
using cuminlp::SlotKind;
using cuminlp::dag::Expr;
using cuminlp::dag::IntervalGraphReplay;
using cuminlp::dag::PointGraphReplay;
using cuminlp::dag::Problem;
using cuminlp::dag::VarKind;

namespace
{

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

// x, y both variables (not constants), objective pow(x, y) -- exercises
// Op::Pow's general a^b, wired via wire_binary<PowOp> onto
// cu::pow(interval, interval).
Problem<double> make_pow_problem(double x_lb, double x_ub, double y_lb, double y_ub)
{
  Problem<double> p;
  auto x = p.var(x_lb, x_ub);
  auto y = p.var(y_lb, y_ub);
  p.set_objective(pow(x, y));
  return p;
}

}  // namespace

static_assert(!std::is_copy_constructible_v<PointGraphReplay<double>>);
static_assert(!std::is_copy_assignable_v<PointGraphReplay<double>>);
static_assert(std::is_move_constructible_v<PointGraphReplay<double>>);

TEST_CASE(
    "GraphReplay::n_regions matches composition_fan_out for its Composition",
    "[graph_replay][5e]")
{
  Problem<double> p = make_problem();
  Composition comp {.kinds = {SlotKind::Continuous}};

  auto replay = IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4});
  CHECK(replay.n_regions() == composition_fan_out(comp, FanOutSpec {4}));
  CHECK(replay.n_regions() == 4);
}

TEST_CASE(
    "set_domain rejects a domain whose size doesn't match the problem's "
    "variable count",
    "[graph_replay][5]")
{
  Problem<double> p = make_problem();
  Composition comp {.kinds = {SlotKind::Continuous}};
  auto replay = IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4});

  std::vector<cu::interval<double>> wrong_size_domain = {{0.0, 10.0},
                                                         {0.0, 1.0}};
  std::vector<std::size_t> var_ids = {0};

  REQUIRE_THROWS_AS(replay.set_domain(wrong_size_domain, var_ids),
                    ShapeMismatch);
}

TEST_CASE("A moved-from GraphReplay is not launchable", "[graph_replay][5]")
{
  Problem<double> p = make_problem();
  Composition comp {.kinds = {SlotKind::Continuous}};
  auto replay = IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4});

  auto moved_to = std::move(replay);

  std::vector<cu::interval<double>> domain = {{0.0, 10.0}};
  std::vector<std::size_t> var_ids = {0};

  // The moved-to object is still perfectly usable.
  REQUIRE_NOTHROW(moved_to.set_domain(domain, var_ids));
  REQUIRE_NOTHROW(moved_to.launch(0));

  // The moved-from object must throw rather than dereference a null device
  // pointer.
  CHECK_THROWS_AS(replay.set_domain(domain, var_ids), cuminlp::error);
  CHECK_THROWS_AS(replay.launch(0), cuminlp::error);
}

TEST_CASE(
    "has_candidate(), candidate_point()'s NaN-ness, and feasibility agree",
    "[graph_replay][5c]")
{
  Composition comp {.kinds = {SlotKind::Continuous}};
  std::vector<std::size_t> var_ids = {0};

  SECTION("a domain with a feasible point has a non-NaN candidate")
  {
    Problem<double> p = make_problem();
    auto replay = PointGraphReplay<double>::build(p, comp, FanOutSpec {4}, 0, 16);

    std::vector<cu::interval<double>> domain = {
        {0.0, 5.0}};  // entirely feasible (x <= 5)
    replay.set_domain(domain, var_ids, /*salt=*/1);
    replay.launch(0);

    REQUIRE(replay.has_candidate());
    auto point = replay.candidate_point();
    REQUIRE(point.size() == 1);
    CHECK(!std::isnan(point[0]));
    CHECK(replay.candidate() < std::numeric_limits<double>::max());
  }

  SECTION(
      "an entirely infeasible domain has no candidate, and its witness is "
      "all-NaN")
  {
    Problem<double> p = make_infeasible_problem();
    auto replay = PointGraphReplay<double>::build(p, comp, FanOutSpec {4}, 0, 16);

    std::vector<cu::interval<double>> domain = {{0.0, 10.0}};
    replay.set_domain(domain, var_ids, /*salt=*/1);
    replay.launch(0);

    CHECK_FALSE(replay.has_candidate());
    auto point = replay.candidate_point();
    REQUIRE(point.size() == 1);
    CHECK(std::isnan(point[0]));
  }
}

TEST_CASE(
    "Replay determinism: identical (domain, var_ids, salt) gives "
    "bitwise-identical results",
    "[graph_replay][5d]")
{
  Problem<double> p = make_problem();
  Composition comp {.kinds = {SlotKind::Continuous}};
  auto replay = PointGraphReplay<double>::build(p, comp, FanOutSpec {4}, 0, 32);

  std::vector<cu::interval<double>> domain = {{0.0, 5.0}};
  std::vector<std::size_t> var_ids = {0};

  replay.set_domain(domain, var_ids, /*salt=*/7);
  replay.launch(0);
  double const first_candidate = replay.candidate();
  auto const first_point_span = replay.candidate_point();
  std::vector<double> first_point(first_point_span.begin(),
                                  first_point_span.end());

  replay.set_domain(domain, var_ids, /*salt=*/7);
  replay.launch(0);
  double const second_candidate = replay.candidate();
  auto const second_point_span = replay.candidate_point();
  std::vector<double> second_point(second_point_span.begin(),
                                   second_point_span.end());

  CHECK(first_candidate == second_candidate);
  REQUIRE(first_point.size() == second_point.size());
  for (std::size_t i = 0; i < first_point.size(); ++i) {
    CHECK(first_point[i] == second_point[i]);
  }
}

TEST_CASE("Replay determinism: different salts draw different sample points",
          "[graph_replay][5d]")
{
  Problem<double> p = make_problem();
  Composition comp {.kinds = {SlotKind::Continuous}};
  auto replay = PointGraphReplay<double>::build(p, comp, FanOutSpec {4}, 0, 32);

  std::vector<cu::interval<double>> domain = {{0.0, 5.0}};
  std::vector<std::size_t> var_ids = {0};

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

TEST_CASE("The interval graph's result is bitwise-identical regardless of salt",
          "[graph_replay][5d]")
{
  // The interval graph's root kernel takes no salt argument at all -- salt
  // only ever reaches set_domain() as a parameter of the shared API surface
  // (point and interval graphs), and must be silently ignored here.
  Problem<double> p = make_problem();
  Composition comp {.kinds = {SlotKind::Continuous}};
  auto replay = IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4});

  std::vector<cu::interval<double>> domain = {{0.0, 10.0}};
  std::vector<std::size_t> var_ids = {0};

  replay.set_domain(domain, var_ids, /*salt=*/1);
  replay.launch(0);
  std::vector<double> obj_lb_salt1(replay.obj_lb().begin(),
                                   replay.obj_lb().end());

  replay.set_domain(domain, var_ids, /*salt=*/2);
  replay.launch(0);
  std::vector<double> obj_lb_salt2(replay.obj_lb().begin(),
                                   replay.obj_lb().end());

  REQUIRE(obj_lb_salt1.size() == obj_lb_salt2.size());
  for (std::size_t i = 0; i < obj_lb_salt1.size(); ++i) {
    CHECK(obj_lb_salt1[i] == obj_lb_salt2[i]);
  }
}

TEST_CASE("Op::Pow: pow(x, y) on a degenerate box matches std::pow exactly",
          "[graph_replay][pow]")
{
  // A single-point box (x=2, y=3) removes any interval-vs-point rounding
  // question -- both graphs must agree with plain std::pow(2, 3) == 8.
  Problem<double> p = make_pow_problem(2.0, 2.0, 3.0, 3.0);
  Composition comp {.kinds = {SlotKind::Continuous, SlotKind::Continuous}};
  std::vector<std::size_t> var_ids = {0, 1};
  std::vector<cu::interval<double>> domain = {{2.0, 2.0}, {3.0, 3.0}};

  auto point_replay = PointGraphReplay<double>::build(p, comp, FanOutSpec {4}, 0, 8);
  point_replay.set_domain(domain, var_ids);
  point_replay.launch(0);
  REQUIRE(point_replay.has_candidate());
  CHECK(point_replay.candidate() == std::pow(2.0, 3.0));

  // The interval graph's directed-rounding enclosure may be a few ulps wide
  // even on a degenerate box (cu::pow rounds its result outward), so this
  // one needs a tolerance rather than bitwise equality.
  auto interval_replay = IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4});
  interval_replay.set_domain(domain, var_ids);
  interval_replay.launch(0);
  REQUIRE(interval_replay.has_candidate());
  CHECK(std::abs(interval_replay.candidate() - std::pow(2.0, 3.0)) < 1e-9);
}

TEST_CASE("Op::Pow: interval graph soundly encloses x^y over a real box",
          "[graph_replay][pow]")
{
  // x in [1, 4], y in [2, 2] (a fixed exponent): the true range of x^2 over
  // [1, 4] is [1, 16]. Regardless of how many sub-regions the partitioner
  // splits [1, 4] into, every per-region lower bound obj_lb() reports must
  // be a sound enclosure of its sub-box (so within [0, 16] overall), and the
  // tightest one must be at most 1 -- the sub-box touching x=1.
  Problem<double> p = make_pow_problem(1.0, 4.0, 2.0, 2.0);
  Composition comp {.kinds = {SlotKind::Continuous, SlotKind::Continuous}};
  std::vector<std::size_t> var_ids = {0, 1};
  std::vector<cu::interval<double>> domain = {{1.0, 4.0}, {2.0, 2.0}};

  auto replay = IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4});
  replay.set_domain(domain, var_ids);
  replay.launch(0);

  REQUIRE(!replay.obj_lb().empty());
  double min_lb = std::numeric_limits<double>::infinity();
  for (double lb : replay.obj_lb()) {
    CHECK(lb >= 0.0 - 1e-9);
    CHECK(lb <= 16.0 + 1e-9);
    min_lb = std::min(min_lb, lb);
  }
  CHECK(min_lb <= 1.0 + 1e-9);
}

// --- Device-memory budget --------------------------------------------------
//
// With runtime fan-out widths, `--partition-num=10` at 20 slots asks for
// 10^20 regions from a command line, and the failure has to be a
// diagnosable exception rather than a partway-through
// cudaErrorMemoryAllocation (or, worse, a wrapped size_t sizing every buffer
// too small).

TEST_CASE("build rejects a composition that exceeds its device-memory budget",
          "[graph_replay][5][config]")
{
  Problem<double> p = make_problem();
  Composition comp {.kinds = {SlotKind::Continuous}};

  // 4 regions of a handful of nodes cannot fit in 16 bytes.
  CHECK_THROWS_AS(
      (IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4}, 16)),
      cuminlp::ResourceExhausted);

  // The same build succeeds under a budget that accommodates it, so the
  // rejection above is the budget talking and not an unrelated failure.
  CHECK_NOTHROW(
      (IntervalGraphReplay<double>::build(p, comp, FanOutSpec {4}, 1u << 30)));
}

TEST_CASE("build rejects a saturated fan-out before allocating",
          "[graph_replay][5][config]")
{
  Problem<double> p = make_problem();

  // 8 continuous slots at partition_num 10^7: composition_fan_out saturates
  // at SIZE_MAX, and estimate_bytes must carry that through rather than
  // wrapping to something that looks affordable.
  Composition huge {.kinds = std::vector<SlotKind>(8, SlotKind::Continuous)};

  CHECK(cuminlp::composition_fan_out(huge, FanOutSpec {10000000})
        == std::numeric_limits<std::size_t>::max());
  CHECK(IntervalGraphReplay<double>::estimate_bytes(
            p, std::numeric_limits<std::size_t>::max())
        == std::numeric_limits<std::size_t>::max());
  CHECK_THROWS_AS(
      (IntervalGraphReplay<double>::build(p, huge, FanOutSpec {10000000})),
      cuminlp::ResourceExhausted);
}

TEST_CASE("estimate_bytes counts only DAG nodes a root actually reaches",
          "[graph_replay][5][config]")
{
  // GraphBuilder allocates lazily from the objective/constraint roots, so a
  // node nothing reaches never gets a buffer. Counting every non-Const node
  // would over-estimate, and over-estimating is the dangerous direction: the
  // budget guard would refuse configurations that would in fact have fit.
  Problem<double> p = make_problem();
  std::size_t const reachable = IntervalGraphReplay<double>::
      count_buffer_nodes(p);

  // Add an expression no root refers to.
  auto dead = p.var(0.0, 1.0) * p.var(0.0, 1.0);
  (void)dead;

  CHECK(IntervalGraphReplay<double>::count_buffer_nodes(p) == reachable);
  CHECK(reachable < p.graph.nodes.size());
}

TEST_CASE("An over-budget build reports the cause and a cap that would fit",
          "[graph_replay][5][config]")
{
  // The raw byte figure alone is not actionable: the region count is a
  // product over slots, so an over-budget request is usually out by orders
  // of magnitude. The report has to name the multiplication and do the
  // arithmetic for the caller.
  Problem<double> p = make_problem();
  Composition comp {.kinds = std::vector<SlotKind>(4, SlotKind::Continuous)};

  FanOutSpec const fan_out {50};  // 50^4 = 6.25M regions

  // The backend throws the facts and the resolver writes the report
  // (design/MODULE_REFACTOR.md §5.6), so the assertions below are on
  // explain_over_budget's output -- the same text, by its new author.
  std::string what;
  std::string summary;
  try {
    IntervalGraphReplay<double>::build(p, comp, fan_out, /*budget=*/100000);
    FAIL("expected OverBudgetError");
  } catch (const cuminlp::backend::OverBudgetError& e) {
    what = cuminlp::explain_over_budget(e.facts(), cuminlp::profile_problem(p));
    summary = e.what();
  }

  INFO(what);
  CHECK(what.find("interval graph") != std::string::npos);  // which graph
  CHECK(what.find("Continuous") != std::string::npos);  // which slot kind
  CHECK(what.find("fan-out 50") != std::string::npos);  // and its width
  CHECK(what.find("6,250,000 regions") != std::string::npos);  // the product
  CHECK(what.find("--max-cycle-size=") != std::string::npos);  // the knob
  CHECK(what.find("--partition-num") != std::string::npos);  // the other knob

  // what() alone stays factual: it names the role and the two byte figures,
  // and nothing that needed a resolver to compute.
  CHECK(summary.find("interval graph needs") != std::string::npos);
  CHECK(summary.find("--max-cycle-size=") == std::string::npos);
  CHECK(what.rfind(summary.substr(0, summary.size() - 1), 0) == 0);
}

TEST_CASE("The suggested cap is one the budget actually admits",
          "[graph_replay][5][config]")
{
  // A suggestion that still doesn't fit would be worse than none. Extract the
  // number the report recommends and check a build at that cap succeeds.
  //
  // Checking only the graph that failed is what let a broken suggestion ship:
  // GraphDriver holds a point, an interval and (when enumerable) an exact
  // graph for the same composition at once, so the report has to be costed
  // against all of them. Every kind is rebuilt below, and the point graph at
  // the *solve-wide* sample count -- the one that is not 1 -- is the case
  // that used to be missed.
  Problem<double> p = make_problem();
  Composition comp {.kinds = std::vector<SlotKind>(4, SlotKind::Continuous)};
  FanOutSpec const fan_out {50};

  std::size_t const budget = 50u * 1024 * 1024;  // 50 MiB
  std::size_t const solve_sample_points = 10;
  std::string what;
  try {
    IntervalGraphReplay<double>::build(
        p, comp, fan_out, budget, solve_sample_points);
    FAIL("expected OverBudgetError");
  } catch (const cuminlp::backend::OverBudgetError& e) {
    what = cuminlp::explain_over_budget(e.facts(), cuminlp::profile_problem(p));
  }

  auto const pos = what.find("--max-cycle-size=");
  REQUIRE(pos != std::string::npos);
  std::size_t const suggested =
      std::stoul(what.substr(pos + std::string("--max-cycle-size=").size()));
  REQUIRE(suggested >= 1);
  REQUIRE(suggested < comp.size());

  // Same composition truncated to the suggested number of live slots -- a
  // composition is exactly its live slots now, so "truncated" just means
  // fewer entries, not a padded-out tail.
  Composition narrowed {.kinds = std::vector<SlotKind>(suggested, SlotKind::Continuous)};

  INFO(what);
  CHECK_NOTHROW((IntervalGraphReplay<double>::build(
      p, narrowed, fan_out, budget, solve_sample_points)));
  CHECK_NOTHROW((PointGraphReplay<double>::build(
      p, narrowed, fan_out, budget, solve_sample_points)));

  // And the combined figure the report quotes is the one the budget has to
  // admit, not any single graph's share of it.
  CHECK(cuminlp::backend::graph::cost_model_for<double>(p).bundle_bytes(
            composition_fan_out(narrowed, fan_out),
            solve_sample_points,
            cuminlp::is_fully_enumerable(narrowed))
        <= budget);
}

TEST_CASE("auto_max_cycle_size picks a cap whose whole graph set fits",
          "[graph_replay][5][config]")
{
  // The CLI's default for --max-cycle-size. Two properties matter and are
  // both checked against composition_footprint_bytes rather than against a
  // hardcoded number: the cap it returns fits, and one slot wider does not.
  // Anything weaker would admit both a cap that OOMs (the bug this replaced)
  // and a cap of 1 that never OOMs but wastes the device.
  Problem<double> p;
  Expr<double> sum = p.int_var(3.0, 9.0);
  for (int i = 1; i < 10; ++i) {
    sum = sum + p.int_var(3.0, 9.0);
  }
  p.set_objective(sum);

  FanOutSpec const fan_out {7};
  std::size_t const sample_points = 5;
  std::size_t const budget = 512u * 1024 * 1024;  // 512 MiB
  auto const cost = cuminlp::backend::graph::cost_model_for<double>(p);

  std::size_t const cap = cuminlp::dag::auto_max_cycle_size(
      /*n_binary=*/0, /*n_integer=*/10, /*n_continuous=*/0, cost, fan_out,
      sample_points, budget, cuminlp::kMaxSlots);
  REQUIRE(cap >= 1);
  REQUIRE(cap <= 10);  // never more slots than the problem has variables

  // Every integer here has domain 7 == enumerate_cap, so the region count at
  // `cap` slots is exactly 7^cap and the composition is fully enumerable.
  auto regions = [](std::size_t slots)
  {
    std::size_t r = 1;
    for (std::size_t i = 0; i < slots; ++i) {
      r *= 7;
    }
    return r;
  };
  CHECK(cost.bundle_bytes(regions(cap), sample_points, true) <= budget);
  if (cap < 10) {
    CHECK(cost.bundle_bytes(regions(cap + 1), sample_points, true) > budget);
  }
}

TEST_CASE("auto_max_cycle_size charges the widest composition the search can "
          "reach, not the root's",
          "[graph_replay][5][config]")
{
  // batch.gms in miniature: binaries fill the root's slots at fan-out 2, so a
  // scan charging slots in the order GreedyCompositionPolicy fills them sees
  // 2^cap and certifies a huge cap. But binaries *resolve* as the search
  // descends, and a descendant fills those same slots with continuous
  // variables at partition_num each -- on batch.gms itself, 2^14 regions at
  // the root against 2^10 * 64^4 = 17.2e9 ten levels down, which is where the
  // certified cap turned into 338.9 TiB of point graph.
  Problem<double> p;
  Expr<double> sum = p.bin_var();
  for (int i = 1; i < 24; ++i) {
    sum = sum + p.bin_var();
  }
  for (int i = 0; i < 22; ++i) {
    sum = sum + p.var(0.0, 10.0);
  }
  p.set_objective(sum);

  FanOutSpec const fan_out {64};
  std::size_t const sample_points = 5;
  std::size_t const budget = 512u * 1024 * 1024;  // 512 MiB
  auto const cost = cuminlp::backend::graph::cost_model_for<double>(p);

  std::size_t const cap = cuminlp::dag::auto_max_cycle_size(
      /*n_binary=*/24, /*n_integer=*/0, /*n_continuous=*/22, cost, fan_out,
      sample_points, budget, cuminlp::kMaxSlots);
  REQUIRE(cap >= 1);

  // The all-continuous composition of `cap` slots is reachable -- every
  // binary resolves eventually -- so it, not the all-binary one, is what the
  // cap has to admit.
  auto continuous_regions = [&](std::size_t slots)
  {
    std::size_t r = 1;
    for (std::size_t i = 0; i < slots; ++i) {
      r *= 64;
    }
    return r;
  };
  CHECK(cost.bundle_bytes(continuous_regions(cap), sample_points, false)
        <= budget);
  CHECK(cost.bundle_bytes(continuous_regions(cap + 1), sample_points, false)
        > budget);

  // And the all-binary composition, which is what the old scan charged, is
  // orders of magnitude cheaper at that same cap -- i.e. the two really do
  // disagree here, so the check above is not vacuous.
  std::size_t binary_regions = 1;
  for (std::size_t i = 0; i < cap; ++i) {
    binary_regions *= 2;
  }
  CHECK(binary_regions < continuous_regions(cap));
}

TEST_CASE("auto_max_cycle_size never exceeds kMaxSlots",
          "[graph_replay][5][config]")
{
  // A budget nothing can exhaust must still return a cap the search-shape
  // ceiling admits.
  Problem<double> p;
  Expr<double> sum = p.var(0.0, 1.0);
  for (int i = 1; i < 200; ++i) {
    sum = sum + p.var(0.0, 1.0);
  }
  p.set_objective(sum);

  std::size_t const cap = cuminlp::dag::auto_max_cycle_size(
      /*n_binary=*/0,
      /*n_integer=*/0,
      /*n_continuous=*/200,
      cuminlp::backend::graph::cost_model_for<double>(p),
      FanOutSpec {2},
      /*sample_points=*/1,
      std::numeric_limits<std::size_t>::max(),
      cuminlp::kMaxSlots);
  CHECK(cap == cuminlp::kMaxSlots);
}
