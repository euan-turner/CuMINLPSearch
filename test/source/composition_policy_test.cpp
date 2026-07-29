#include <algorithm>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"

using cuminlp::can_fathom_without_children;
using cuminlp::Composition;
using cuminlp::GreedyCompositionPolicy;
using cuminlp::ShapeMismatch;
using cuminlp::SlotKind;
using cuminlp::dag::VarKind;

TEST_CASE("GreedyCompositionPolicy fills binary slots before integer slots", "[composition_policy]")
{
  // vars: 0 continuous [0,10], 1 binary [0,1], 2 integer [0,3] (4 values)
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0},
      {0.0, 1.0},
      {0.0, 3.0},
  };
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  // PartitionNum = 4: the integer's domain size (4) is enumerable.
  GreedyCompositionPolicy<double, 2, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::BinaryEnumerate);
  CHECK(assignment.var_ids[1] == 2);
  CHECK(assignment.composition[1] == SlotKind::IntegerEnumerate);
}

TEST_CASE("GreedyCompositionPolicy bisects integer domains above PartitionNum", "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  GreedyCompositionPolicy<double, 1, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.composition[0] == SlotKind::IntegerBisect);
}

TEST_CASE("GreedyCompositionPolicy's EnumerateCap is independent of PartitionNum's bisection width",
         "[composition_policy]")
{
  // Domain size 40 -- not enumerable under PartitionNum (4) alone, but is
  // under an EnumerateCap (50) that doesn't also force continuous/bisected
  // slots to a width of 50.
  std::vector<cu::interval<double>> box = {{0.0, 39.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Continuous};

  GreedyCompositionPolicy<double, 2, /*PartitionNum=*/4, /*EnumerateCap=*/50> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
  CHECK(assignment.var_ids[1] == 1);
  CHECK(assignment.composition[1] == SlotKind::Continuous);

  // The continuous slot still bisects at PartitionNum (4), not EnumerateCap.
  CHECK(cuminlp::slot_fan_out<4, 50>(SlotKind::Continuous) == 4);
  CHECK(cuminlp::slot_fan_out<4, 50>(SlotKind::IntegerEnumerate) == 50);
}

TEST_CASE("GreedyCompositionPolicy prefers the smallest remaining integer domain", "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}, {0.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Integer};

  GreedyCompositionPolicy<double, 1, 50> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
}

TEST_CASE("GreedyCompositionPolicy prefers the widest continuous variable", "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 2.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};

  GreedyCompositionPolicy<double, 1, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
}

TEST_CASE("GreedyCompositionPolicy spills into the next kind once one runs out", "[composition_policy]")
{
  // Only one binary available, but two slots to fill; second slot should
  // fall through to the integer.
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 5.0}};
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 2, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.var_ids[1] == 1);
}

TEST_CASE("GreedyCompositionPolicy pads with Padding (fan-out 1) once every variable is resolved",
         "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyCompositionPolicy<double, 2, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.var_ids[1] == 0);
  // Padding, not Continuous: a resolved box's Composition must stay fully
  // enumerable so it can still be fathomed (TEST_EXTENSION.md §4b).
  CHECK(assignment.composition[0] == SlotKind::Padding);
  CHECK(assignment.composition[1] == SlotKind::Padding);
  CHECK(cuminlp::slot_fan_out<4>(SlotKind::Padding) == 1);
}

TEST_CASE("GreedyCompositionPolicy::choose is a pure function of (box, var_kinds)", "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0},
      {0.0, 1.0},
      {0.0, 3.0},
  };
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 2, 4> policy;
  auto first = policy.choose(box, kinds);
  auto second = policy.choose(box, kinds);

  CHECK(first.composition == second.composition);
  CHECK(first.var_ids == second.var_ids);
}

TEST_CASE("GreedyCompositionPolicy::choose throws ShapeMismatch on an empty var_kinds span",
         "[composition_policy]")
{
  // A zero-variable box has no valid var_id to pad slots with (0 isn't < 0);
  // Problem::validate() now rejects zero-variable Problems too, but the
  // policy's own contract should reject this directly rather than silently
  // writing an out-of-range var_id (TEST_EXTENSION.md §3).
  std::vector<cu::interval<double>> box;
  std::vector<VarKind> kinds;

  GreedyCompositionPolicy<double, 2, 4> policy;
  REQUIRE_THROWS_AS(policy.choose(box, kinds), ShapeMismatch);
}

TEST_CASE("integer_domain_size does not underflow on a sub-box with no integer point",
         "[composition_policy][3b]")
{
  using Policy = GreedyCompositionPolicy<double, 1, 50>;

  // A normal, lattice-aligned domain still counts correctly.
  CHECK(Policy::integer_domain_size({0.0, 3.0}) == 4);

  // Reachable directly from an IntegerBisect child: ceil(lb) > floor(ub), no
  // integer point at all. Must classify as 0 (empty), not wrap to a huge
  // size_t via an out-of-range double->size_t cast.
  CHECK(Policy::integer_domain_size({2.5, 2.9}) == 0);
  CHECK(Policy::integer_domain_size({2.5, 2.9}) < 1000);
}

TEST_CASE("A sub-box with no integer point is still classified as enumerable rather than "
         "bisected forever",
         "[composition_policy][3b]")
{
  // Under the old integer_domain_size, {2.5, 2.9}'s domain size underflowed
  // to a huge value, comparing as > EnumerateCap and driving another bisect
  // forever. With the fix, its (empty) domain size of 0 is <= any
  // EnumerateCap, so it terminates as IntegerEnumerate instead.
  std::vector<cu::interval<double>> box = {{2.5, 2.9}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  GreedyCompositionPolicy<double, 1, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
}

// TEST_EXTENSION.md §4b: the fathoming predicate, tested directly and
// independent of the surrounding search loop (see graph_driver.cuh).
TEST_CASE("can_fathom_without_children at live_count == CycleSize", "[composition_policy][4b]")
{
  Composition<3> comp = {SlotKind::BinaryEnumerate, SlotKind::BinaryEnumerate, SlotKind::BinaryEnumerate};
  CHECK(can_fathom_without_children(3, comp));
}

TEST_CASE("can_fathom_without_children at live_count == CycleSize - 1 (the padding case)",
         "[composition_policy][4b]")
{
  // 2 live binaries under CycleSize == 3: before the Padding fix, the
  // trailing slot was blanket-Continuous, making is_fully_enumerable false
  // and permanently denying this box the exact/fathom path despite being
  // trivially enumerable.
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Binary};

  GreedyCompositionPolicy<double, 3, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.composition[2] == SlotKind::Padding);
  CHECK(can_fathom_without_children(2, assignment.composition));
}

TEST_CASE("can_fathom_without_children at live_count == 0 (fully resolved box)",
         "[composition_policy][4b]")
{
  // A fully-resolved box must be fathomed, never partitioned -- otherwise it
  // re-enqueues copies of itself indefinitely (TEST_EXTENSION.md §4b).
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyCompositionPolicy<double, 2, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(can_fathom_without_children(0, assignment.composition));
}

TEST_CASE("can_fathom_without_children is false when a live Continuous slot remains",
         "[composition_policy][4b]")
{
  Composition<2> comp = {SlotKind::BinaryEnumerate, SlotKind::Continuous};
  CHECK_FALSE(can_fathom_without_children(2, comp));
}

TEST_CASE("GreedyCompositionPolicy::possible_compositions enumerates every reachable single-slot composition",
         "[composition_policy]")
{
  // At least one of each kind present, so none of the five structural
  // patterns get capped away by var-kind counts. With CycleSize == 1 the
  // one slot is either BinaryEnumerate, IntegerEnumerate, IntegerBisect,
  // real Continuous, or unused Padding -- 5 possibilities, not 4: Padding is
  // distinct from a genuine Continuous slot (TEST_EXTENSION.md §4b), even
  // though both were the same SlotKind::Continuous before that fix.
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 1, 4> policy;
  auto compositions = policy.possible_compositions(kinds);

  auto contains = [&](SlotKind kind) {
    return std::any_of(compositions.begin(), compositions.end(),
                       [&](const Composition<1>& c) { return c[0] == kind; });
  };

  CHECK(compositions.size() == 5);
  CHECK(contains(SlotKind::Continuous));
  CHECK(contains(SlotKind::IntegerBisect));
  CHECK(contains(SlotKind::IntegerEnumerate));
  CHECK(contains(SlotKind::BinaryEnumerate));
  CHECK(contains(SlotKind::Padding));
}

TEST_CASE("GreedyCompositionPolicy::choose always returns a Composition possible_compositions() reports",
         "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0},
      {0.0, 1.0},
      {0.0, 3.0},
  };
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 2, 4> policy;
  auto assignment = policy.choose(box, kinds);
  auto compositions = policy.possible_compositions(kinds);

  CHECK(std::find(compositions.begin(), compositions.end(), assignment.composition)
        != compositions.end());
}

TEST_CASE("GreedyCompositionPolicy::possible_compositions caps k/m by the problem's actual variable counts",
         "[composition_policy]")
{
  // 3 binary variables, no integer or continuous ones. Uncapped, CycleSize=5
  // would yield 56 structurally-possible splits of binary/integer/continuous
  // slot counts; capped by what this problem can actually produce (m is
  // always 0 -- there are no integers to ever fill an integer slot, so k can
  // only run 0..3), there are exactly 4: k = 0, 1, 2, 3.
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Binary, VarKind::Binary};

  GreedyCompositionPolicy<double, 5, 4> policy;
  auto compositions = policy.possible_compositions(kinds);

  CHECK(compositions.size() == 4);
  for (const auto& c : compositions) {
    for (SlotKind kind : c) {
      CHECK(kind != SlotKind::IntegerEnumerate);
      CHECK(kind != SlotKind::IntegerBisect);
    }
  }
}
