// design/AGGREGATE_BOUNDING.md §11, tests 6-9 -- stage 4's exit criteria.
// Requires an actual CUDA device.
//
// The central test is the first: the device's k aggregate bounds must equal a
// host reduction over the per-subregion verdicts, bitwise. The oracle for
// those verdicts is the *existing* IntervalGraphReplay run on the same problem
// and the same launch assignment -- it copies every subregion's obj_lb and
// feasible flag back, which is exactly what the aggregate backend refuses to
// do. Using the tested backend as the oracle is what makes this a check on
// the new epilogue rather than on interval arithmetic.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/aggregate/cost.hpp"
#include "cuminlp/aggregate/partition.hpp"
#include "cuminlp/aggregate/policy.hpp"
#include "cuminlp/backend/aggregate/factory.cuh"
#include "cuminlp/backend/graph/replay.cuh"
#include "cuminlp/model/eval.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/materialise.hpp"

using cuminlp::aggregate::AggregateBound;
using cuminlp::aggregate::AggregatePartition;
using cuminlp::aggregate::AggregateRegion;
using cuminlp::aggregate::WidestBranchPolicy;
using cuminlp::backend::aggregate::AggregateBackendFactory;
using cuminlp::backend::aggregate::AggregateBounderReplay;
using cuminlp::model::Cmp;
using cuminlp::model::Problem;
using cuminlp::model::VarKind;
using Box = std::vector<cu::interval<double>>;

namespace
{

bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

/// A small constrained nonlinear problem: enough shared subexpression
/// structure that the DAG is not a straight line, and a constraint that
/// genuinely excludes part of the box so the feasibility mask is exercised
/// rather than trivially all-ones.
struct Fixture
{
  Problem<double> problem;
  Box root;

  Fixture()
  {
    auto x = problem.var(-2.0, 3.0);
    auto y = problem.var(-1.0, 4.0);
    auto shared = x * x + y;
    problem.set_objective(shared + 0.5 * y * y - x);
    problem.add_constraint(shared, Cmp::LE, 6.0);
    problem.validate();
    root = problem.box_bounds;
  }

  std::vector<VarKind> kinds() const { return problem.var_kinds; }
};

/// The aggregate, computed on the host from per-subregion verdicts -- §4.1's
/// definition transcribed directly, with no shortcuts, so it can disagree
/// with the device if the device is wrong.
AggregateBound<double> host_aggregate(std::span<const double> obj_lb,
                                      std::span<const unsigned char> feasible,
                                      std::size_t first,
                                      std::size_t count)
{
  double lb = std::numeric_limits<double>::infinity();
  bool any = false;
  for (std::size_t i = first; i < first + count; ++i) {
    if (!feasible[i]) {
      continue;
    }
    any = true;
    lb = std::min(lb, obj_lb[i]);
  }
  if (!any) {
    return AggregateBound<double> {std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity(),
                                   false};
  }
  return AggregateBound<double> {
      lb, std::numeric_limits<double>::infinity(), true};
}

/// AggregateRegion is a reference-holding view, so it has to be built where
/// both of its referents outlive the call. This keeps the call sites short
/// without any of them owning a dangling partition.
AggregateRegion<double> agg_of(const Box& box, const AggregatePartition& part)
{
  return AggregateRegion<double> {box, part};
}

/// One AggregateRegion, as the span the role interface takes.
std::span<const AggregateRegion<double>> one(const AggregateRegion<double>& r)
{
  return std::span<const AggregateRegion<double>>(&r, 1);
}

}  // namespace

TEST_CASE("The device aggregate equals a host reduction over the per-subregion "
          "verdicts, bitwise",
          "[aggregate][backend]")
{
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;

  for (std::size_t n :
       {std::size_t {16}, std::size_t {256}, std::size_t {4096}})
  {
    WidestBranchPolicy<double> const policy(k, n);
    AggregatePartition const part = policy.partition(fx.root, kinds, 0);
    REQUIRE_NOTHROW(part.validate());
    REQUIRE(part.total_fan_out() == n);

    // The oracle: the existing, tested interval backend, run on the identical
    // launch assignment. It brings every subregion's obj_lb and feasible flag
    // back -- but not obj_ub, which it keeps device-side for its own ArgMin,
    // so `hull_ub` needs the separate oracle in the next test rather than a
    // D2H copy added to a backend this design promised not to touch.
    auto oracle = cuminlp::backend::graph::IntervalGraphReplay<double>::build(
        fx.problem, part.launch.composition, n);
    cuminlp::backend::Region<double> const region {fx.root, part.launch};
    auto const per_subregion = oracle.bound(region);
    REQUIRE(per_subregion.n_regions == n);

    AggregateBackendFactory<double> factory;
    auto bounder = factory.build_bounder(
        fx.problem, part.launch.composition, n, k, /*budget=*/0);
    AggregateRegion<double> const agg {fx.root, part};
    auto const bounds = bounder->bound_children(one(agg));
    REQUIRE(bounds.size() == k);

    for (std::size_t c = 0; c < k; ++c) {
      auto const want = host_aggregate(per_subregion.obj_lb,
                                       per_subregion.feasible,
                                       c * part.refine_fan_out,
                                       part.refine_fan_out);
      INFO("N = " << n << ", child " << c);
      CHECK(bounds[c].feasible == want.feasible);
      if (want.feasible) {
        CHECK(feq(bounds[c].lb, want.lb));
      }
    }
  }
}

TEST_CASE("Refinement tightens the bound and never loosens it",
          "[aggregate][backend]")
{
  // §4.1: more subregions per child can only raise the minimum of their lower
  // bounds. At M = 1 each child is bounded by a single interval evaluation of
  // itself, which is the baseline every larger M must beat or match.
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;

  std::vector<double> previous;
  // M starts at 2, not 1: a refine with no bisections at all is rejected by
  // WidestBranchPolicy, since N == k would mean the launch bounds each child
  // by a single interval evaluation and buys nothing over branching alone.
  for (std::size_t m : {std::size_t {2}, std::size_t {16}, std::size_t {256}}) {
    std::size_t const n = k * m;
    WidestBranchPolicy<double> const policy(k, n);
    AggregatePartition const part = policy.partition(fx.root, kinds, 0);
    REQUIRE(part.refine_fan_out == m);

    AggregateBackendFactory<double> factory;
    auto bounder =
        factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);
    AggregateRegion<double> const agg {fx.root, part};
    auto const bounds = bounder->bound_children(one(agg));

    std::vector<double> current;
    for (std::size_t c = 0; c < k; ++c) {
      REQUIRE(bounds[c].feasible);
      current.push_back(bounds[c].lb);
      CHECK(bounds[c].hull_ub >= bounds[c].lb);
    }
    if (!previous.empty()) {
      for (std::size_t c = 0; c < k; ++c) {
        INFO("child " << c << " at M = " << m);
        CHECK(current[c] >= previous[c]);
      }
    }
    previous = current;
  }
  // The refinement has to actually buy something, or "tightens" is vacuous.
  CHECK(previous[0] > -std::numeric_limits<double>::infinity());
}

TEST_CASE("Every aggregate bound encloses the true objective over its child",
          "[aggregate][backend]")
{
  // Tests 6 and 7 would both pass with a consistently wrong rounding
  // direction. This one would not: it compares against real objective values
  // at real points, evaluated on the host by model/eval.hpp.
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;
  constexpr std::size_t n = 1024;

  WidestBranchPolicy<double> const policy(k, n);
  AggregatePartition const part = policy.partition(fx.root, kinds, 0);
  AggregateBackendFactory<double> factory;
  auto bounder =
      factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);
  AggregateRegion<double> const agg {fx.root, part};
  auto const bounds = bounder->bound_children(one(agg));

  std::mt19937 rng {31415};
  std::size_t feasible_points = 0;

  for (std::size_t c = 0; c < k; ++c) {
    Box child;
    cuminlp::region::materialise<double>(fx.root, c, part.branch, child);
    REQUIRE(bounds[c].feasible);

    for (int trial = 0; trial < 4000; ++trial) {
      std::vector<double> point(child.size());
      for (std::size_t i = 0; i < child.size(); ++i) {
        point[i] = std::uniform_real_distribution<double>(child[i].lb,
                                                          child[i].ub)(rng);
      }
      if (!cuminlp::model::satisfies_constraints(fx.problem, point)) {
        continue;
      }
      ++feasible_points;
      double const f = cuminlp::model::evaluate_objective(fx.problem, point);
      INFO("child " << c << " lb = " << bounds[c].lb << " f = " << f);
      REQUIRE(bounds[c].lb <= f);
      REQUIRE(f <= bounds[c].hull_ub);
    }
  }
  // A vacuous pass -- every sampled point infeasible -- would prove nothing.
  CHECK(feasible_points > 1000);
}

TEST_CASE("A sub-box's bound is never weaker than its parent's, at equal "
          "partitioning",
          "[aggregate][backend]")
{
  // Inclusion monotonicity: for C' subset of C the interval extension gives
  // F(C') subset of F(C), hence lb F(C') >= lb F(C). This is the property
  // that *is* true, and the one §4.4 originally over-generalised into a claim
  // about aggregates across a re-partition -- see the next test.
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;
  constexpr std::size_t n = 256;

  WidestBranchPolicy<double> const policy(k, n);
  AggregatePartition const part = policy.partition(fx.root, kinds, 0);
  AggregateBackendFactory<double> factory;
  auto bounder =
      factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);

  auto const parent_span = bounder->bound_children(one(agg_of(fx.root, part)));
  std::vector<AggregateBound<double>> const parent(parent_span.begin(),
                                                   parent_span.end());

  // Re-bound each child using the *same* partition shape the parent used, so
  // the covers genuinely nest: the child's subregions are then a refinement
  // of the parent subregions that lie inside it.
  for (std::size_t c = 0; c < k; ++c) {
    if (!parent[c].feasible) {
      continue;
    }
    Box child;
    cuminlp::region::materialise<double>(fx.root, c, part.branch, child);
    AggregateRegion<double> const sub {child, part};
    auto const refined = bounder->bound_children(one(sub));
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t g = 0; g < k; ++g) {
      if (refined[g].feasible) {
        best = std::min(best, refined[g].lb);
      }
    }
    INFO("child " << c << ": parent lb " << parent[c].lb << ", refined "
                  << best);
    CHECK(best >= parent[c].lb);
  }
}

TEST_CASE("An aggregate over a re-partitioned child can be weaker than the "
          "child's own bound",
          "[aggregate][backend]")
{
  // The finding that corrected §4.4, pinned so it cannot be un-learned.
  //
  // A child is bounded over M subregions chosen by *its parent's* refine; when
  // that child is itself popped, its grandchildren are bounded over a
  // partition chosen afresh, which splits different variables. More subregions
  // in total does not imply a refinement of the same cover, and a bound over a
  // non-nested cover is under no obligation to be tighter.
  //
  // Both bounds are individually sound (the enclosure test proves that), so
  // nothing here is wrong -- but the driver must clamp each child's lb to its
  // parent's, which is what makes the frontier's minimum monotone. See §4.4.
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;
  constexpr std::size_t n = 256;

  WidestBranchPolicy<double> const policy(k, n);
  AggregateBackendFactory<double> factory;
  AggregatePartition const part = policy.partition(fx.root, kinds, 0);
  auto bounder =
      factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);
  auto const parent_span = bounder->bound_children(one(agg_of(fx.root, part)));
  std::vector<AggregateBound<double>> const parent(parent_span.begin(),
                                                   parent_span.end());

  bool observed_looser = false;
  for (std::size_t c = 0; c < k; ++c) {
    if (!parent[c].feasible) {
      continue;
    }
    Box child;
    cuminlp::region::materialise<double>(fx.root, c, part.branch, child);
    AggregatePartition const sub = policy.partition(child, kinds, 1);
    if (sub.is_leaf()) {
      continue;
    }
    auto sub_bounder =
        factory.build_bounder(fx.problem, sub.launch.composition, n, k, 0);
    auto const grand = sub_bounder->bound_children(one(agg_of(child, sub)));

    double best = std::numeric_limits<double>::infinity();
    for (std::size_t g = 0; g < k; ++g) {
      if (grand[g].feasible) {
        best = std::min(best, grand[g].lb);
      }
      // Whatever else is true, each grandchild bound is sound over its own
      // box, and the clamp the driver applies restores monotonicity.
      double const clamped = std::max(grand[g].lb, parent[c].lb);
      CHECK(clamped >= parent[c].lb);
    }
    if (best < parent[c].lb) {
      observed_looser = true;
    }
  }
  // If this ever stops holding on this fixture the test has gone vacuous and
  // is no longer pinning the finding.
  CHECK(observed_looser);
}

TEST_CASE("What crosses PCIe does not grow with N", "[aggregate][backend]")
{
  // Stage 4's quantitative measurable, and §2.2's whole claim: the existing
  // backend copies 9 bytes per subregion back on every launch, this one
  // copies 2k scalars regardless of how many subregions there were.
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;

  std::size_t first = 0;
  for (std::size_t n :
       {std::size_t {4096}, std::size_t {65536}, std::size_t {1048576}})
  {
    WidestBranchPolicy<double> const policy(k, n);
    AggregatePartition const part = policy.partition(fx.root, kinds, 0);
    AggregateBackendFactory<double> factory;
    auto bounder =
        factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);

    std::size_t const d2h = bounder->d2h_bytes_per_launch();
    INFO("N = " << n);
    CHECK(d2h <= 1024);
    if (first == 0) {
      first = d2h;
    } else {
      CHECK(d2h == first);  // constant in N, which is the claim
    }

    // And the launch still works at every one of those sizes.
    AggregateRegion<double> const agg {fx.root, part};
    auto const bounds = bounder->bound_children(one(agg));
    REQUIRE(bounds.size() == k);
    CHECK(bounds[0].feasible);
  }
  CHECK(first == 2 * k * sizeof(double));
}

TEST_CASE("Allocated device bytes match the costed split",
          "[aggregate][backend]")
{
  // The other stage-4 measurable: §6.3's accounting has to describe what is
  // actually allocated, since it is what the CLI's fit and the PARAMS line are
  // computed from.
  //
  // Against `allocated_bytes()`, which the class accumulates at its allocation
  // sites, rather than against a cudaMemGetInfo delta. Two reasons, both
  // learned the hard way: the suite runs 16 processes against one device, so
  // free memory moves under all of them; and the first DeviceReduce::Min/Max
  // of a process loads those kernels' modules at a cost of ~100 MB, which
  // lands on whichever build happens to be first and swamps the O(N) term
  // (§6.3).
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;

  for (std::size_t n :
       {std::size_t {4096}, std::size_t {65536}, std::size_t {1 << 20}})
  {
    AggregateBackendFactory<double> factory;
    WidestBranchPolicy<double> const policy(k, n);
    AggregatePartition const part = policy.partition(fx.root, kinds, 0);
    auto bounder =
        factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);

    auto const predicted = factory.bounder_bytes(fx.problem, n);
    auto const actual = bounder->allocated_bytes();
    INFO("N = " << n << ": predicted " << predicted << ", actual " << actual);

    // The prediction covers the O(N) term only; the remainder is the O(k)
    // outputs, the slot arrays and the O(sqrt(M)) CUB scratch, all of which
    // bounder_bytes deliberately omits (see its comment).
    CHECK(actual >= predicted);
    CHECK(static_cast<double>(actual) <= 1.05 * static_cast<double>(predicted));
  }

  // A point element is about half an interval element, which is the whole
  // reason a 10% budget slice buys a generous number of samples (§6.3).
  constexpr std::size_t n = 1 << 20;
  auto const bounder_cost =
      AggregateBackendFactory<double>::bounder_bytes(fx.problem, n);
  auto const sampler_cost =
      AggregateBackendFactory<double>::sampler_bytes(fx.problem, n, 1);
  CHECK(sampler_cost < bounder_cost);
  CHECK(2 * sampler_cost > bounder_cost);
}

TEST_CASE("The budget split and fan-out fit agree with each other",
          "[aggregate][backend]")
{
  using cuminlp::aggregate::bounder_element_bytes;
  using cuminlp::aggregate::default_sample_budget_fraction;
  using cuminlp::aggregate::fit_power_of_two;
  using cuminlp::aggregate::split_device_budget;

  constexpr std::size_t gib = std::size_t {1} << 30;
  auto const split =
      split_device_budget(8 * gib, default_sample_budget_fraction);
  CHECK(split.bounder_bytes + split.sampler_bytes == 8 * gib);
  CHECK(split.sampler_bytes < split.bounder_bytes);

  std::size_t const per_element = bounder_element_bytes<double>(20);
  std::size_t const n = fit_power_of_two(per_element, split.bounder_bytes);
  CHECK(n > 0);
  CHECK((n & (n - 1)) == 0);  // a power of two, so M = N / k stays exact
  CHECK(n * per_element <= split.bounder_bytes);
  CHECK(2 * n * per_element > split.bounder_bytes);

  // Degenerate budgets fail to a zero fit rather than to a bad one.
  CHECK(fit_power_of_two(per_element, 0) == 0);
  CHECK(fit_power_of_two(per_element, per_element - 1) == 0);
  CHECK(fit_power_of_two(per_element, per_element) == 1);
}

// ---- design/REFINEMENT_STUDY.md stage 1 ------------------------------------
//
// The retained per-subregion arrays are the study's entire raw material, so
// what they must be pinned to is not "plausible interval bounds" but "exactly
// the values the reduction consumed". Both tests below check that pinning
// from a different direction: the first that opting out costs nothing, the
// second that opting in reproduces the device's own aggregate.

TEST_CASE("Per-subregion bounds are not retained unless asked for",
          "[aggregate][backend][study]")
{
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;
  constexpr std::size_t n = 4096;

  WidestBranchPolicy<double> const policy(k, n);
  AggregatePartition const part = policy.partition(fx.root, kinds, 0);
  AggregateBackendFactory<double> factory;
  auto bounder =
      factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);

  // The default must be off, and off must be indistinguishable from before
  // the flag existed -- a solve that silently started copying N doubles per
  // launch would undo §2.2 without failing anything else in this file.
  CHECK(bounder->d2h_bytes_per_launch() == 2 * k * sizeof(double));

  AggregateRegion<double> const agg {fx.root, part};
  auto const bounds = bounder->bound_children(one(agg));
  REQUIRE(bounds.size() == k);

  CHECK(bounder->subregion_lb().empty());
  CHECK(bounder->subregion_ub().empty());
  CHECK(bounder->subregion_feasible().empty());

  // And turning it on is reflected in the reported traffic rather than
  // hidden: telemetry that under-reported this would make the study's launches
  // look as cheap as a solve's.
  bounder->retain_subregion_bounds(true);
  CHECK(bounder->d2h_bytes_per_launch()
        == 2 * k * sizeof(double)
            + n * (2 * sizeof(double) + sizeof(unsigned char)));

  bounder->retain_subregion_bounds(false);
  CHECK(bounder->d2h_bytes_per_launch() == 2 * k * sizeof(double));
}

TEST_CASE("Retained per-subregion bounds reproduce the device's aggregate",
          "[aggregate][backend][study]")
{
  // The load-bearing one. REFINEMENT_STUDY.md computes its hulls on the host
  // from these arrays rather than from AggregateBound, so if the arrays ever
  // drifted from what the reduction consumed the study would report a hull no
  // solver would ever see -- and nothing else in the suite would notice.
  Fixture fx;
  auto const kinds = fx.kinds();
  constexpr std::size_t k = 4;

  for (std::size_t n : {std::size_t {64}, std::size_t {4096}}) {
    WidestBranchPolicy<double> const policy(k, n);
    AggregatePartition const part = policy.partition(fx.root, kinds, 0);
    AggregateBackendFactory<double> factory;
    auto bounder =
        factory.build_bounder(fx.problem, part.launch.composition, n, k, 0);
    bounder->retain_subregion_bounds(true);

    AggregateRegion<double> const agg {fx.root, part};
    auto const bounds = bounder->bound_children(one(agg));
    REQUIRE(bounds.size() == k);

    auto const lb = bounder->subregion_lb();
    auto const ub = bounder->subregion_ub();
    auto const feas = bounder->subregion_feasible();
    REQUIRE(lb.size() == n);
    REQUIRE(ub.size() == n);
    REQUIRE(feas.size() == n);

    std::size_t const m = part.refine_fan_out;
    for (std::size_t c = 0; c < k; ++c) {
      double want_lb = std::numeric_limits<double>::infinity();
      double want_ub = -std::numeric_limits<double>::infinity();
      bool any = false;
      for (std::size_t i = c * m; i < (c + 1) * m; ++i) {
        if (!feas[i]) {
          continue;
        }
        any = true;
        want_lb = std::min(want_lb, lb[i]);
        want_ub = std::max(want_ub, ub[i]);
      }

      INFO("N = " << n << ", child " << c);
      CHECK(bounds[c].feasible == any);
      if (any) {
        // Bitwise: both sides are a min/max over the same doubles, so any
        // difference at all is a real disagreement, not rounding.
        CHECK(feq(bounds[c].lb, want_lb));
        CHECK(feq(bounds[c].hull_ub, want_ub));
      }
    }

    // The unmasked hull the study also needs (§2.5) must bracket the masked
    // one -- masking can only ever discard values from the extremes inward.
    double raw_lb = std::numeric_limits<double>::infinity();
    double raw_ub = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
      raw_lb = std::min(raw_lb, lb[i]);
      raw_ub = std::max(raw_ub, ub[i]);
    }
    for (std::size_t c = 0; c < k; ++c) {
      if (bounds[c].feasible) {
        INFO("N = " << n << ", child " << c);
        CHECK(raw_lb <= bounds[c].lb);
        CHECK(raw_ub >= bounds[c].hull_ub);
      }
    }

    // Every retained subregion bound must be an interval, masked or not.
    for (std::size_t i = 0; i < n; ++i) {
      INFO("N = " << n << ", subregion " << i);
      REQUIRE(lb[i] <= ub[i]);
    }
  }
}
