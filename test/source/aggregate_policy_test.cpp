// design/AGGREGATE_BOUNDING.md §11, tests 1 and 2 -- stage 2's exit criteria.
//
// Host-only: aggregate/partition.hpp and aggregate/policy.hpp name
// region/, policy/ and model/ and no CUDA type, so every invariant §3.2 and
// §5 rest on is decidable without a GPU. That is deliberate -- the layout
// assumption "child_id = r / M" is the one the device epilogue is built
// around, and it is far cheaper to break it here than there.

#include <cstddef>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/aggregate/partition.hpp"
#include "cuminlp/aggregate/policy.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/materialise.hpp"

using cuminlp::aggregate::AggregatePartition;
using cuminlp::aggregate::child_of_region;
using cuminlp::aggregate::WidestBranchPolicy;
using cuminlp::model::VarKind;
using cuminlp::region::composition_fan_out;
using Box = std::vector<cu::interval<double>>;
using Kinds = std::vector<VarKind>;

namespace
{

/// `inner` sits inside `outer` on every dimension. The containment check is
/// non-strict and un-tolerated: the branch and the launch decode the same
/// variable through the same `slot_bounds` arithmetic, so a child's bounds
/// are expected to be exactly representable relative to its parent's, not
/// merely close. A tolerance here would hide exactly the off-by-one-part
/// layout error the test exists to catch.
bool contains(const Box& outer, const Box& inner)
{
  if (outer.size() != inner.size()) {
    return false;
  }
  for (std::size_t i = 0; i < outer.size(); ++i) {
    if (inner[i].lb < outer[i].lb || inner[i].ub > outer[i].ub) {
      return false;
    }
  }
  return true;
}

/// Bitwise equality spelled with two `<`, so this TU can keep
/// -Werror=float-equal on (same technique as dag_test.cpp's `feq`). The
/// checks below want exact agreement, not approximate: the branch decode and
/// the launch decode run the identical `slot_bounds` arithmetic on the
/// identical inputs, so anything short of bitwise equality is a real
/// divergence.
bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

std::size_t live_count(const Box& box)
{
  std::size_t live = 0;
  for (const auto& b : box) {
    if (b.ub > b.lb) {
      ++live;
    }
  }
  return live;
}

/// Randomised (box, var_kinds) pairs across the shapes that stress §5.4: a
/// varying number of live dimensions including 0, 1 and 2, mixed kinds, and
/// widths spanning several orders of magnitude so the branch's choice of
/// variable actually moves.
struct BoxGenerator
{
  std::mt19937 rng {20260811};

  std::pair<Box, Kinds> next()
  {
    std::uniform_int_distribution<std::size_t> n_vars_dist(1, 8);
    std::uniform_int_distribution<int> kind_dist(0, 2);
    std::uniform_int_distribution<int> dead_dist(0, 3);  // 1 in 4 resolved
    std::uniform_real_distribution<double> mag_dist(-3.0, 4.0);
    std::uniform_real_distribution<double> lo_dist(-50.0, 50.0);

    std::size_t const n = n_vars_dist(rng);
    Box box(n);
    Kinds kinds(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto const kind = static_cast<VarKind>(kind_dist(rng));
      kinds[i] = kind;
      if (kind == VarKind::Binary) {
        box[i] = {0.0, dead_dist(rng) == 0 ? 0.0 : 1.0};
        continue;
      }
      double lo = lo_dist(rng);
      double width = std::pow(10.0, mag_dist(rng));
      if (kind == VarKind::Integer) {
        lo = std::round(lo);
        width = std::max(1.0, std::round(width));
      }
      box[i] = {lo, dead_dist(rng) == 0 ? lo : lo + width};
    }
    return {box, kinds};
  }
};

}  // namespace

TEST_CASE("A partition satisfies every invariant it claims", "[aggregate]")
{
  // Small N so the exhaustive per-region sweep in the next test stays cheap;
  // the invariants are independent of how large N is.
  WidestBranchPolicy<double> const policy(4, 64);
  BoxGenerator gen;

  std::size_t leaves = 0;
  std::size_t nested = 0;
  std::size_t disjoint = 0;

  for (int trial = 0; trial < 1200; ++trial) {
    auto const [box, kinds] = gen.next();
    AggregatePartition const p = policy.partition(box, kinds, 0);

    // validate() is the statement of the invariants; that it does not throw
    // is the assertion. Checking the products again below is not redundant --
    // it is what catches a validate() that silently checks nothing.
    REQUIRE_NOTHROW(p.validate());

    if (live_count(box) == 0) {
      CHECK(p.is_leaf());
      ++leaves;
      continue;
    }

    REQUIRE_FALSE(p.is_leaf());
    CHECK(composition_fan_out(p.branch.widths) == 4);
    CHECK(composition_fan_out(p.launch.widths) == 64);

    if (p.nested) {
      ++nested;
      CHECK(live_count(box) == 1);
      CHECK(p.launch.size() == 1);
      CHECK(p.refine.size() == 0);
    } else {
      ++disjoint;
      CHECK(live_count(box) >= 2);
      CHECK(composition_fan_out(p.refine.widths) == 16);
      CHECK(p.launch.size() == p.branch.size() + p.refine.size());
      for (std::size_t bv : p.branch.var_ids) {
        CHECK(std::find(p.refine.var_ids.begin(), p.refine.var_ids.end(), bv)
              == p.refine.var_ids.end());
      }
    }
  }

  // The generator has to actually reach all three shapes, or the loop above
  // is asserting over a corpus that never exercises §5.4's hard case.
  CHECK(leaves > 0);
  CHECK(nested > 0);
  CHECK(disjoint > 0);
}

TEST_CASE("Every launch subregion sits inside the branch child that "
          "child_id = r / M assigns it to",
          "[aggregate]")
{
  // §3.2's layout property, and the one the device epilogue is built around:
  // if this fails, each child is reduced over some other child's subregions
  // and every bound is attributed to the wrong box.
  WidestBranchPolicy<double> const policy(4, 64);
  BoxGenerator gen;

  std::size_t checked = 0;
  for (int trial = 0; trial < 1200; ++trial) {
    auto const [box, kinds] = gen.next();
    AggregatePartition const p = policy.partition(box, kinds, 0);
    if (p.is_leaf()) {
      continue;
    }

    // The k child boxes, decoded the way the host driver decodes them.
    std::vector<Box> children(p.branch_fan_out);
    for (std::size_t c = 0; c < p.branch_fan_out; ++c) {
      cuminlp::region::materialise<double>(box, c, p.branch, children[c]);
      REQUIRE(contains(box, children[c]));
    }

    // Every subregion the device will evaluate, decoded the way
    // apply_slots_kernel decodes it.
    for (std::size_t r = 0; r < p.total_fan_out(); ++r) {
      Box sub;
      cuminlp::region::materialise<double>(box, r, p.launch, sub);
      std::size_t const c = child_of_region(r, p.refine_fan_out);
      REQUIRE(c < p.branch_fan_out);
      REQUIRE(contains(children[c], sub));
      ++checked;
    }
  }
  CHECK(checked > 50000);
}

TEST_CASE("branch is a pure function of the box it is given", "[aggregate]")
{
  // The one contract §5.1 keeps from the old CompositionPolicy: a child's box
  // is re-derived by calling branch() again on a reconstructed parent, so two
  // calls on equal boxes must agree exactly.
  WidestBranchPolicy<double> const policy(4, 1024);
  BoxGenerator gen;

  for (int trial = 0; trial < 500; ++trial) {
    auto const [box, kinds] = gen.next();
    auto const a = policy.branch(box, kinds);
    auto const b = policy.branch(box, kinds);
    CHECK(cuminlp::region::assignment_hash(a)
          == cuminlp::region::assignment_hash(b));
  }
}

TEST_CASE("The branch splits the widest live variable, k ways, uniformly",
          "[aggregate]")
{
  WidestBranchPolicy<double> const policy(4, 1024);
  Kinds const kinds {
      VarKind::Continuous, VarKind::Continuous, VarKind::Integer};

  Box const box {{0.0, 1.0}, {0.0, 100.0}, {0.0, 3.0}};
  auto const branch = policy.branch(box, kinds);
  REQUIRE(branch.size() == 1);
  CHECK(branch.var_ids[0] == 1);  // the width-100 dimension
  CHECK(branch.widths[0] == 4);
  CHECK(branch.composition[0] == cuminlp::region::SlotKind::Continuous);

  // A discrete branch variable partitions rather than enumerates -- only a
  // uniform split nests (see WidestBranchPolicy's class comment).
  Box const discrete {{0.0, 1.0}, {0.0, 2.0}, {0.0, 500.0}};
  auto const on_integer = policy.branch(discrete, kinds);
  REQUIRE(on_integer.size() == 1);
  CHECK(on_integer.var_ids[0] == 2);
  CHECK(on_integer.composition[0]
        == cuminlp::region::SlotKind::IntegerPartition);
}

TEST_CASE("A fully resolved box is a leaf, not a partition", "[aggregate]")
{
  WidestBranchPolicy<double> const policy(4, 1024);
  Kinds const kinds {VarKind::Continuous, VarKind::Integer};
  Box const resolved {{1.0, 1.0}, {2.0, 2.0}};

  AggregatePartition const p = policy.partition(resolved, kinds, 0);
  CHECK(p.is_leaf());
  CHECK(p.launch.size() == 0);
  REQUIRE_NOTHROW(p.validate());
}

TEST_CASE("The nested case keeps child_id = r / M exact on one variable",
          "[aggregate]")
{
  // Named rather than left to the randomised sweep: this is §5.4's hard case
  // and the reason the branch takes exactly one variable.
  WidestBranchPolicy<double> const policy(4, 64);
  Kinds const kinds {VarKind::Continuous, VarKind::Continuous};
  Box const box {{0.0, 8.0}, {5.0, 5.0}};  // exactly one live dimension

  AggregatePartition const p = policy.partition(box, kinds, 0);
  REQUIRE(p.nested);
  REQUIRE_NOTHROW(p.validate());
  CHECK(p.launch.widths[0] == 64);

  // The c-th of 4 equal parts must be exactly parts [c*16, (c+1)*16) of the
  // same variable split 64 ways -- the property that makes nesting safe.
  for (std::size_t c = 0; c < 4; ++c) {
    Box child;
    cuminlp::region::materialise<double>(box, c, p.branch, child);
    Box first;
    Box last;
    cuminlp::region::materialise<double>(box, c * 16, p.launch, first);
    cuminlp::region::materialise<double>(box, c * 16 + 15, p.launch, last);
    CHECK(feq(first[0].lb, child[0].lb));
    CHECK(feq(last[0].ub, child[0].ub));
  }
}

TEST_CASE("validate rejects a partition whose branch slots are not the most "
          "significant",
          "[aggregate]")
{
  // The invariant has to be able to fail, or the sweep above proves nothing.
  WidestBranchPolicy<double> const policy(4, 64);
  Kinds const kinds {VarKind::Continuous, VarKind::Continuous};
  Box const box {{0.0, 8.0}, {0.0, 4.0}};

  AggregatePartition p = policy.partition(box, kinds, 0);
  REQUIRE_FALSE(p.nested);
  REQUIRE_NOTHROW(p.validate());

  // Branch slots first instead of last: the product is untouched, so only the
  // ordering clause can catch this.
  p.launch = cuminlp::aggregate::concatenate(p.branch, p.refine);
  CHECK(composition_fan_out(p.launch.widths) == 64);
  CHECK_THROWS_AS(p.validate(), cuminlp::InvalidConfiguration);
}

TEST_CASE("The policy rejects fan-outs it cannot honour", "[aggregate]")
{
  using cuminlp::InvalidConfiguration;
  CHECK_THROWS_AS(WidestBranchPolicy<double>(3, 64), InvalidConfiguration);
  CHECK_THROWS_AS(WidestBranchPolicy<double>(1, 64), InvalidConfiguration);
  CHECK_THROWS_AS(WidestBranchPolicy<double>(4, 4), InvalidConfiguration);
  CHECK_THROWS_AS(WidestBranchPolicy<double>(4, 100), InvalidConfiguration);
  CHECK_NOTHROW(WidestBranchPolicy<double>(4, 8));
}
