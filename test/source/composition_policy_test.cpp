#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"

using cuminlp::Composition;
using cuminlp::GreedyCompositionPolicy;
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

TEST_CASE("GreedyCompositionPolicy pads with a no-op once every variable is resolved", "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyCompositionPolicy<double, 2, 4> policy;
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.var_ids[1] == 0);
}

TEST_CASE("GreedyCompositionPolicy::possible_compositions enumerates every reachable single-slot composition",
         "[composition_policy]")
{
  // At least one of each kind present, so none of the four structural
  // patterns get capped away by var-kind counts.
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 1, 4> policy;
  auto compositions = policy.possible_compositions(kinds);

  auto contains = [&](SlotKind kind) {
    return std::any_of(compositions.begin(), compositions.end(),
                       [&](const Composition<1>& c) { return c[0] == kind; });
  };

  CHECK(compositions.size() == 4);
  CHECK(contains(SlotKind::Continuous));
  CHECK(contains(SlotKind::IntegerBisect));
  CHECK(contains(SlotKind::IntegerEnumerate));
  CHECK(contains(SlotKind::BinaryEnumerate));
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
