// design/REFINEMENT_STUDY.md stage 2: the N=1 device baseline, the masked and
// unmasked hull reductions over the retained per-subregion bounds, and the Q1
// gain metrics.
//
// The load-bearing test here is soundness against the point evaluator. There
// is no host interval arithmetic to check the device against (§3.3: cuinterval
// is __device__-only, and model/eval.hpp evaluates at points, not over boxes),
// so the oracle is the one independent evaluator that exists. It cannot catch
// a bound that is too loose -- which is what this study measures, not a bug --
// but it does catch one that is too tight, which would silently corrupt every
// number the study reports.
//
// Requires an actual CUDA device.

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/backend/aggregate/replay.cuh"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/model/eval.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/bisection_budget.hpp"
#include "cuminlp/study/parent_box.hpp"
#include "cuminlp/study/refinement.hpp"

using cuminlp::backend::aggregate::AggregateBounderReplay;
using cuminlp::model::Cmp;
using cuminlp::model::Problem;
using cuminlp::model::VarKind;
using cuminlp::policy::BisectionBudgetCompositionPolicy;
using cuminlp::region::Composition;
using cuminlp::region::SlotAssignment;
using cuminlp::study::Hull;
using Box = std::vector<cu::interval<double>>;

namespace
{

/// Two problems rather than one: a constrained instance so the feasibility
/// mask is exercised, and an unconstrained one so the masked and unmasked
/// hulls must coincide (which is itself a check that masking is doing what
/// §2.5 claims and not something else).
struct Constrained
{
  Problem<double> problem;
  Box root;

  Constrained()
  {
    auto x = problem.var(-2.0, 3.0);
    auto y = problem.var(-1.0, 4.0);
    auto shared = x * x + y;
    problem.set_objective(shared + 0.5 * y * y - x);
    problem.add_constraint(shared, Cmp::LE, 6.0);
    problem.validate();
    root = problem.box_bounds;
  }
};

struct Rosenbrock
{
  Problem<double> problem;
  Box root;

  Rosenbrock()
  {
    auto x = problem.var(-2.0, 2.0);
    auto y = problem.var(-1.0, 3.0);
    auto a = 1.0 - x;
    auto b = y - x * x;
    problem.set_objective(a * a + 100.0 * (b * b));
    problem.validate();
    root = problem.box_bounds;
  }
};

/// §3.3's baseline: one region, no slots, so the single "subregion" is the
/// parent box itself and its bound is plain interval analysis over it.
Hull<double> device_baseline(const Problem<double>& problem, const Box& box)
{
  Composition const empty {};
  auto bounder = AggregateBounderReplay<double>::build(
      problem, empty, /*n_regions=*/1, /*branch_fan_out=*/1, /*budget=*/0);
  bounder.retain_subregion_bounds(true);
  bounder.set_domain(box, {}, {});
  bounder.launch(/*stream=*/0);
  return cuminlp::study::unmasked_hull<double>(bounder.subregion_lb(),
                                               bounder.subregion_ub());
}

/// One launch at bisection budget B, returning the retained arrays by value so
/// the caller can reduce them however it likes after the bounder dies.
struct Refined
{
  std::vector<double> lb;
  std::vector<double> ub;
  std::vector<unsigned char> feasible;
};

Refined device_refined(const Problem<double>& problem,
                       const Box& box,
                       std::size_t budget)
{
  BisectionBudgetCompositionPolicy<double> const policy(budget);
  SlotAssignment const a = policy.choose(box, problem.var_kinds);
  std::size_t const n = cuminlp::region::composition_fan_out(a.widths);

  auto bounder = AggregateBounderReplay<double>::build(
      problem, a.composition, n, /*branch_fan_out=*/1, /*budget=*/0);
  bounder.retain_subregion_bounds(true);
  bounder.set_domain(box, a.var_ids, a.widths);
  bounder.launch(/*stream=*/0);

  Refined out;
  auto lb = bounder.subregion_lb();
  auto ub = bounder.subregion_ub();
  auto fe = bounder.subregion_feasible();
  out.lb.assign(lb.begin(), lb.end());
  out.ub.assign(ub.begin(), ub.end());
  out.feasible.assign(fe.begin(), fe.end());
  return out;
}

/// Points drawn uniformly from `box`, for the soundness oracle. Seeded, so a
/// failure is reproducible.
std::vector<std::vector<double>> sample_box(const Box& box,
                                            std::size_t count,
                                            std::uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<std::vector<double>> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<double> p(box.size());
    for (std::size_t v = 0; v < box.size(); ++v) {
      std::uniform_real_distribution<double> d(box[v].lb, box[v].ub);
      p[v] = d(rng);
    }
    out.push_back(std::move(p));
  }
  return out;
}

}  // namespace

TEST_CASE("The N=1 baseline is plain interval analysis over the parent box",
          "[study][refinement]")
{
  Constrained fx;
  Hull<double> const base = device_baseline(fx.problem, fx.root);
  REQUIRE(base.any);

  // Hand-computed, and worth spelling out because it pins the dependency
  // problem this study exists to measure. The DSL lowers `x * x` to Op::Sqr,
  // giving [0, 9] rather than mul([-2,3], [-2,3]) = [-6, 9]; but `0.5 * y * y`
  // parses as `(0.5 * y) * y`, a product of two *different* expressions, so it
  // gets [-0.5, 2] * [-1, 4] = [-2, 8] rather than 0.5 * sqr(y) = [0, 8].
  //   lb = 0 + (-1) + (-2) + (-3) = -6
  //   ub = 9 +   4  +   8  +   2  = 23
  CHECK(base.lb == -6.0);
  CHECK(base.ub == 23.0);
}

TEST_CASE("Every bound encloses the objective at points of its box",
          "[study][refinement]")
{
  // §3.3's soundness oracle. A bound that is too tight fails here; a bound
  // that is merely loose does not, and should not.
  Constrained cfx;
  Rosenbrock rfx;

  struct Case
  {
    const char* name;
    const Problem<double>* problem;
    const Box* box;
  };
  Case const cases[] = {{"constrained", &cfx.problem, &cfx.root},
                        {"rosenbrock", &rfx.problem, &rfx.root}};

  for (const Case& c : cases) {
    Hull<double> const base = device_baseline(*c.problem, *c.box);
    REQUIRE(base.any);

    for (std::size_t budget : {std::size_t {1}, std::size_t {6}}) {
      Refined const r = device_refined(*c.problem, *c.box, budget);
      Hull<double> const unmasked =
          cuminlp::study::unmasked_hull<double>(r.lb, r.ub);
      Hull<double> const masked =
          cuminlp::study::masked_hull<double>(r.lb, r.ub, r.feasible);
      REQUIRE(unmasked.any);

      INFO(c.name << ", budget " << budget);

      // (a) the refined hull is contained in the baseline
      CHECK(unmasked.lb >= base.lb);
      CHECK(unmasked.ub <= base.ub);

      // (c) masking only ever discards values inward
      if (masked.any) {
        CHECK(masked.lb >= unmasked.lb);
        CHECK(masked.ub <= unmasked.ub);
      }

      // (b) soundness against the point evaluator. The unmasked hull must
      // enclose f at *every* point of the box. The masked hull must not be
      // checked this way: it legitimately excludes parts of the box, so a
      // point in an excluded subregion may well fall outside it.
      for (const auto& p : sample_box(*c.box, 200, 20260812u)) {
        double const f =
            cuminlp::model::evaluate_objective(*c.problem, p);
        INFO("f = " << f << " vs baseline [" << base.lb << ", " << base.ub
                    << "] and hull [" << unmasked.lb << ", " << unmasked.ub
                    << "]");
        REQUIRE(f >= base.lb);
        REQUIRE(f <= base.ub);
        REQUIRE(f >= unmasked.lb);
        REQUIRE(f <= unmasked.ub);
      }
    }
  }
}

TEST_CASE("An unconstrained problem's masked and unmasked hulls coincide",
          "[study][refinement]")
{
  // If these ever diverged, masking would be responding to something other
  // than constraints, and §2.5's attribution of the difference to constraint
  // propagation would be wrong.
  Rosenbrock fx;
  Refined const r = device_refined(fx.problem, fx.root, 8);
  auto const unmasked = cuminlp::study::unmasked_hull<double>(r.lb, r.ub);
  auto const masked =
      cuminlp::study::masked_hull<double>(r.lb, r.ub, r.feasible);

  REQUIRE(masked.any);
  CHECK(cuminlp::study::excluded_fraction(r.feasible) == 0.0);
  CHECK(masked.lb == unmasked.lb);
  CHECK(masked.ub == unmasked.ub);
}

TEST_CASE("A constrained problem's mask genuinely excludes subregions",
          "[study][refinement]")
{
  // The converse guard: if the constraint excluded nothing, the previous
  // test would pass trivially on both fixtures and neither would be checking
  // masking at all.
  Constrained fx;
  Refined const r = device_refined(fx.problem, fx.root, 8);
  CHECK(cuminlp::study::excluded_fraction(r.feasible) > 0.0);

  auto const unmasked = cuminlp::study::unmasked_hull<double>(r.lb, r.ub);
  auto const masked =
      cuminlp::study::masked_hull<double>(r.lb, r.ub, r.feasible);
  REQUIRE(masked.any);
  CHECK(masked.width() <= unmasked.width());
}

TEST_CASE("Gain metrics are non-negative and degrade gracefully",
          "[study][refinement]")
{
  using cuminlp::study::gains;

  SECTION("a refined hull inside a baseline yields non-negative gains")
  {
    Hull<double> const base {-10.0, 10.0, true};
    Hull<double> const refined {-4.0, 6.0, true};
    auto const g = gains(base, refined);
    CHECK(g.dual_gain == 6.0);
    CHECK(g.primal_side_gain == 4.0);
    REQUIRE(g.width_ratio.has_value());
    CHECK(*g.width_ratio == 0.5);
  }

  SECTION("a constant objective gives no ratio rather than a NaN")
  {
    // (d): U0 == L0. Dividing would give 0/0; the metric is undefined, and
    // saying so is the only honest option -- a NaN here would average into a
    // summary silently.
    Hull<double> const base {3.0, 3.0, true};
    Hull<double> const refined {3.0, 3.0, true};
    auto const g = gains(base, refined);
    CHECK_FALSE(g.width_ratio.has_value());
    CHECK(g.dual_gain == 0.0);
  }

  SECTION("an unbounded baseline gives no ratio")
  {
    Hull<double> const base {-std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity(),
                             true};
    Hull<double> const refined {-5.0, 5.0, true};
    auto const g = gains(base, refined);
    CHECK_FALSE(g.width_ratio.has_value());
    // Refinement making a vacuous bound finite is a real and interesting
    // result (§2.1), so the gain itself is still reported, as +inf.
    CHECK(std::isinf(g.dual_gain));
  }

  SECTION("an empty hull yields zeroed gains rather than garbage")
  {
    Hull<double> const base {-10.0, 10.0, true};
    Hull<double> const empty {};
    auto const g = gains(base, empty);
    CHECK(g.dual_gain == 0.0);
    CHECK_FALSE(g.width_ratio.has_value());
    CHECK(std::isinf(empty.width()));
  }
}

// ---- stage 3: the parent-box family and the refinement ladder -------------

TEST_CASE("The sigma ladder is geometric and widest-first",
          "[study][refinement]")
{
  auto const l = cuminlp::study::sigma_ladder(4);
  REQUIRE(l.size() == 5);
  CHECK(l[0] == 1.0);
  CHECK(l[1] == 0.5);
  CHECK(l[4] == 0.0625);
  for (std::size_t i = 1; i < l.size(); ++i) {
    CHECK(l[i] < l[i - 1]);
  }

  // sigma = 1 admits exactly one placement: the root. Drawing `reps` copies
  // would report duplicate launches as independent samples (§3.1).
  CHECK(cuminlp::study::placements_for(1.0, 8) == 1);
  CHECK(cuminlp::study::placements_for(0.5, 8) == 8);
}

TEST_CASE("Drawn parent boxes are reproducible and inside the root",
          "[study][refinement]")
{
  Constrained fx;
  auto const kinds = fx.problem.var_kinds;

  SECTION("sigma = 1 returns the root exactly and consumes no randomness")
  {
    std::mt19937_64 rng(1234);
    auto const b =
        cuminlp::study::draw_parent_box<double>(fx.root, kinds, 1.0, rng);
    REQUIRE(b.size() == fx.root.size());
    for (std::size_t v = 0; v < b.size(); ++v) {
      CHECK(b[v].lb == fx.root[v].lb);
      CHECK(b[v].ub == fx.root[v].ub);
    }
    CHECK(rng == std::mt19937_64(1234));  // untouched
  }

  SECTION("the same seed reproduces the same boxes")
  {
    for (double sigma : {0.5, 0.25, 0.0625}) {
      std::mt19937_64 a(20260812), b(20260812);
      auto const box_a =
          cuminlp::study::draw_parent_box<double>(fx.root, kinds, sigma, a);
      auto const box_b =
          cuminlp::study::draw_parent_box<double>(fx.root, kinds, sigma, b);
      for (std::size_t v = 0; v < box_a.size(); ++v) {
        INFO("sigma " << sigma << " var " << v);
        CHECK(box_a[v].lb == box_b[v].lb);
        CHECK(box_a[v].ub == box_b[v].ub);
      }
    }
  }

  SECTION("every draw is contained in the root, at every sigma")
  {
    std::mt19937_64 rng(99);
    for (double sigma : cuminlp::study::sigma_ladder(10)) {
      for (int rep = 0; rep < 50; ++rep) {
        auto const box =
            cuminlp::study::draw_parent_box<double>(fx.root, kinds, sigma, rng);
        INFO("sigma = " << sigma << ", rep " << rep);
        REQUIRE(cuminlp::study::inside<double>(box, fx.root));
      }
    }
  }

  SECTION("achieved relative width tracks sigma for continuous variables")
  {
    std::mt19937_64 rng(7);
    for (double sigma : {0.5, 0.25, 0.125}) {
      auto const box =
          cuminlp::study::draw_parent_box<double>(fx.root, kinds, sigma, rng);
      double const got =
          cuminlp::study::achieved_relative_width<double>(box, fx.root);
      INFO("sigma = " << sigma << ", achieved " << got);
      CHECK(std::abs(got - sigma) < 1e-9);
    }
  }
}

TEST_CASE("Discrete variables stay on the integer lattice",
          "[study][refinement]")
{
  // §3.1's snapping rule. A box with fractional endpoints is not a sub-problem
  // the discrete structure can occupy, so shrinking must snap outward -- and
  // outward snapping must not be allowed to leave the root.
  Problem<double> q;
  auto x = q.var(-10.0, 10.0);
  auto k = q.var(-7.0, 7.0, VarKind::Integer);
  q.set_objective(x * x + k);
  q.validate();
  Box const root = q.box_bounds;
  auto const kinds = q.var_kinds;

  std::mt19937_64 rng(4242);
  for (double sigma : cuminlp::study::sigma_ladder(12)) {
    for (int rep = 0; rep < 40; ++rep) {
      auto const box =
          cuminlp::study::draw_parent_box<double>(root, kinds, sigma, rng);
      INFO("sigma = " << sigma << ", rep " << rep);
      REQUIRE(cuminlp::study::inside<double>(box, root));

      // The integer variable's endpoints are integral, and it never empties.
      double const lo = box[1].lb;
      double const hi = box[1].ub;
      REQUIRE(lo == std::floor(lo));
      REQUIRE(hi == std::floor(hi));
      REQUIRE(hi >= lo);
    }
  }
}

TEST_CASE("The hull is monotone along the refinement ladder",
          "[study][refinement]")
{
  // §3.2's nesting invariant, as a check rather than an expectation: the
  // greedy heap's first B pops are a stable prefix of its first B+1, so the
  // budget-(B+1) partition refines the budget-B one and the hull can only
  // tighten. A violation means either a policy bug or an unsound reduction,
  // and either would invalidate every rho(N) curve the study produces.
  Constrained cfx;
  Rosenbrock rfx;

  struct Case
  {
    const char* name;
    const Problem<double>* problem;
    const Box* box;
  };
  Case const cases[] = {{"constrained", &cfx.problem, &cfx.root},
                        {"rosenbrock", &rfx.problem, &rfx.root}};

  for (const Case& c : cases) {
    Hull<double> prev;
    for (std::size_t budget = 1; budget <= 10; ++budget) {
      Refined const r = device_refined(*c.problem, *c.box, budget);
      auto const h = cuminlp::study::unmasked_hull<double>(r.lb, r.ub);
      REQUIRE(h.any);

      INFO(c.name << ", budget " << budget);
      if (prev.any) {
        CHECK(h.lb >= prev.lb);  // dual bound never loosens
        CHECK(h.ub <= prev.ub);  // nor does the hull's top
        CHECK(h.width() <= prev.width());
      }
      prev = h;
    }
  }
}
