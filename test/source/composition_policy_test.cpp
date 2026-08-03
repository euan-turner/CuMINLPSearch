#include <algorithm>
#include <limits>
#include <vector>

#include "cuminlp/composition_policy.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"

using cuminlp::can_fathom_without_children;
using cuminlp::Composition;
using cuminlp::FanOutSpec;
using cuminlp::GreedyCompositionPolicy;
using cuminlp::is_fully_enumerable;
using cuminlp::ShapeMismatch;
using cuminlp::SlotKind;
using cuminlp::dag::VarKind;

TEST_CASE("GreedyCompositionPolicy fills binary slots before integer slots",
          "[composition_policy]")
{
  // vars: 0 continuous [0,10], 1 binary [0,1], 2 integer [0,3] (4 values)
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0},
      {0.0, 1.0},
      {0.0, 3.0},
  };
  std::vector<VarKind> kinds = {
      VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  // PartitionNum = 4: the integer's domain size (4) is enumerable.
  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::BinaryEnumerate);
  CHECK(assignment.var_ids[1] == 2);
  CHECK(assignment.composition[1] == SlotKind::IntegerEnumerate);
}

TEST_CASE("GreedyCompositionPolicy bisects integer domains above PartitionNum",
          "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  GreedyCompositionPolicy<double, 1> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.composition[0] == SlotKind::IntegerBisect);
}

TEST_CASE(
    "GreedyCompositionPolicy's enumerate_cap is independent of partition_num's "
    "bisection width",
    "[composition_policy]")
{
  // Domain size 40 -- not enumerable under PartitionNum (4) alone, but is
  // under an EnumerateCap (50) that doesn't also force continuous/bisected
  // slots to a width of 50.
  std::vector<cu::interval<double>> box = {{0.0, 39.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Continuous};

  GreedyCompositionPolicy<double, 2> policy {
      FanOutSpec {/*partition_num=*/4, /*enumerate_cap=*/50}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
  CHECK(assignment.var_ids[1] == 1);
  CHECK(assignment.composition[1] == SlotKind::Continuous);

  // The continuous slot still bisects at PartitionNum (4), not EnumerateCap.
  CHECK(cuminlp::slot_fan_out(SlotKind::Continuous, FanOutSpec {4, 50}) == 4);
  CHECK(cuminlp::slot_fan_out(SlotKind::IntegerEnumerate, FanOutSpec {4, 50}) == 50);
}

TEST_CASE(
    "GreedyCompositionPolicy prefers the smallest remaining integer domain",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}, {0.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Integer};

  GreedyCompositionPolicy<double, 1> policy {FanOutSpec {50}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
}

TEST_CASE("GreedyCompositionPolicy prefers the widest continuous variable",
          "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 2.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};

  GreedyCompositionPolicy<double, 1> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
}

TEST_CASE("GreedyCompositionPolicy spills into the next kind once one runs out",
          "[composition_policy]")
{
  // Only one binary available, but two slots to fill; second slot should
  // fall through to the integer.
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 5.0}};
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.var_ids[1] == 1);
}

TEST_CASE(
    "GreedyCompositionPolicy pads with Padding (fan-out 1) once every variable "
    "is resolved",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.var_ids[1] == 0);
  // Padding, not Continuous: a resolved box's Composition must stay fully
  // enumerable so it can still be fathomed (TEST_EXTENSION.md).
  CHECK(assignment.composition[0] == SlotKind::Padding);
  CHECK(assignment.composition[1] == SlotKind::Padding);
  CHECK(cuminlp::slot_fan_out(SlotKind::Padding, FanOutSpec {4}) == 1);
}

TEST_CASE(
    "GreedyCompositionPolicy::choose is a pure function of (box, var_kinds)",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0},
      {0.0, 1.0},
      {0.0, 3.0},
  };
  std::vector<VarKind> kinds = {
      VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4}};
  auto first = policy.choose(box, kinds);
  auto second = policy.choose(box, kinds);

  CHECK(first.composition == second.composition);
  CHECK(first.var_ids == second.var_ids);
}

TEST_CASE(
    "GreedyCompositionPolicy::choose throws ShapeMismatch on an empty "
    "var_kinds span",
    "[composition_policy]")
{
  // A zero-variable box has no valid var_id to pad slots with (0 isn't < 0);
  // Problem::validate() now rejects zero-variable Problems too, but the
  // policy's own contract should reject this directly rather than silently
  // writing an out-of-range var_id (TEST_EXTENSION.md).
  std::vector<cu::interval<double>> box;
  std::vector<VarKind> kinds;

  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4}};
  REQUIRE_THROWS_AS(policy.choose(box, kinds), ShapeMismatch);
}

TEST_CASE(
    "integer_domain_size does not underflow on a sub-box with no integer point",
    "[composition_policy][3b]")
{
  using Policy = GreedyCompositionPolicy<double, 1>;

  // A normal, lattice-aligned domain still counts correctly.
  CHECK(Policy::integer_domain_size({0.0, 3.0}) == 4);

  // Reachable directly from an IntegerBisect child: ceil(lb) > floor(ub), no
  // integer point at all. Must classify as 0 (empty), not wrap to a huge
  // size_t via an out-of-range double->size_t cast.
  CHECK(Policy::integer_domain_size({2.5, 2.9}) == 0);
  CHECK(Policy::integer_domain_size({2.5, 2.9}) < 1000);
}

TEST_CASE(
    "A sub-box with no integer point is still classified as enumerable rather "
    "than " "bisected forever",
    "[composition_policy][3b]")
{
  // Under the old integer_domain_size, {2.5, 2.9}'s domain size underflowed
  // to a huge value, comparing as > EnumerateCap and driving another bisect
  // forever. With the fix, its (empty) domain size of 0 is <= any
  // EnumerateCap, so it terminates as IntegerEnumerate instead.
  std::vector<cu::interval<double>> box = {{2.5, 2.9}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  GreedyCompositionPolicy<double, 1> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
}

// TEST_EXTENSION.md: the fathoming predicate, tested directly and
// independent of the surrounding search loop (see graph_driver.cuh).
TEST_CASE("can_fathom_without_children at live_count == slot count",
          "[composition_policy][4b]")
{
  // `count` is what the predicate compares against, not the capacity: a
  // hand-built Composition has to state it, exactly as choose() would.
  Composition<3> comp {.kinds = {SlotKind::BinaryEnumerate,
                                 SlotKind::BinaryEnumerate,
                                 SlotKind::BinaryEnumerate},
                       .count = 3};
  CHECK(can_fathom_without_children(3, comp));
}

TEST_CASE(
    "can_fathom_without_children is false when live variables outnumber the "
    "slots the policy filled",
    "[composition_policy][4b]")
{
  // The case a capped policy on a wider rung creates, and the reason the
  // bound is `count` rather than `Capacity`: three slots are compiled, the
  // policy filled two, and a third variable is live with nowhere to go.
  // Testing against the capacity would call this fathomable and discard a
  // subtree that may hold the optimum.
  Composition<3> comp {
      .kinds = {SlotKind::BinaryEnumerate, SlotKind::BinaryEnumerate,
                SlotKind::Padding},
      .count = 2};
  CHECK(is_fully_enumerable(comp));
  CHECK_FALSE(can_fathom_without_children(3, comp));
  CHECK(can_fathom_without_children(2, comp));
}

TEST_CASE(
    "can_fathom_without_children at live_count == CycleSize - 1 (the padding "
    "case)",
    "[composition_policy][4b]")
{
  // 2 live binaries under CycleSize == 3: before the Padding fix, the
  // trailing slot was blanket-Continuous, making is_fully_enumerable false
  // and permanently denying this box the exact/fathom path despite being
  // trivially enumerable.
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Binary};

  GreedyCompositionPolicy<double, 3> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.composition[2] == SlotKind::Padding);
  CHECK(can_fathom_without_children(2, assignment.composition));
}

TEST_CASE("can_fathom_without_children at live_count == 0 (fully resolved box)",
          "[composition_policy][4b]")
{
  // A fully-resolved box must be fathomed, never partitioned -- otherwise it
  // re-enqueues copies of itself indefinitely (TEST_EXTENSION.md).
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(can_fathom_without_children(0, assignment.composition));
}

TEST_CASE(
    "can_fathom_without_children is false when a live Continuous slot remains",
    "[composition_policy][4b]")
{
  Composition<2> comp {.kinds = {SlotKind::BinaryEnumerate, SlotKind::Continuous},
                       .count = 2};
  CHECK_FALSE(can_fathom_without_children(2, comp));
}

TEST_CASE(
    "GreedyCompositionPolicy::possible_compositions enumerates every reachable "
    "single-slot composition",
    "[composition_policy]")
{
  // At least one of each kind present, so none of the five structural
  // patterns get capped away by var-kind counts. With CycleSize == 1 the
  // one slot is either BinaryEnumerate, IntegerEnumerate, IntegerBisect,
  // real Continuous, or unused Padding -- 5 possibilities, not 4: Padding is
  // distinct from a genuine Continuous slot (TEST_EXTENSION.md), even
  // though both were the same SlotKind::Continuous before that fix.
  std::vector<VarKind> kinds = {
      VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 1> policy {FanOutSpec {4}};
  auto compositions = policy.possible_compositions(kinds);

  auto contains = [&](SlotKind kind)
  {
    return std::any_of(compositions.begin(),
                       compositions.end(),
                       [&](const Composition<1>& c) { return c[0] == kind; });
  };

  CHECK(compositions.size() == 5);
  CHECK(contains(SlotKind::Continuous));
  CHECK(contains(SlotKind::IntegerBisect));
  CHECK(contains(SlotKind::IntegerEnumerate));
  CHECK(contains(SlotKind::BinaryEnumerate));
  CHECK(contains(SlotKind::Padding));
}

TEST_CASE(
    "GreedyCompositionPolicy::choose always returns a Composition "
    "possible_compositions() reports",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0},
      {0.0, 1.0},
      {0.0, 3.0},
  };
  std::vector<VarKind> kinds = {
      VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);
  auto compositions = policy.possible_compositions(kinds);

  CHECK(std::find(
            compositions.begin(), compositions.end(), assignment.composition)
        != compositions.end());
}

TEST_CASE(
    "GreedyCompositionPolicy::possible_compositions caps k/m by the problem's "
    "actual variable counts",
    "[composition_policy]")
{
  // 3 binary variables, no integer or continuous ones. Uncapped, CycleSize=5
  // would yield 56 structurally-possible splits of binary/integer/continuous
  // slot counts; capped by what this problem can actually produce (m is
  // always 0 -- there are no integers to ever fill an integer slot, so k can
  // only run 0..3), there are exactly 4: k = 0, 1, 2, 3.
  std::vector<VarKind> kinds = {
      VarKind::Binary, VarKind::Binary, VarKind::Binary};

  GreedyCompositionPolicy<double, 5> policy {FanOutSpec {4}};
  auto compositions = policy.possible_compositions(kinds);

  CHECK(compositions.size() == 4);
  for (const auto& c : compositions) {
    for (SlotKind kind : c) {
      CHECK(kind != SlotKind::IntegerEnumerate);
      CHECK(kind != SlotKind::IntegerBisect);
    }
  }
}

// --- FanOutSpec: the checks the compiler used to do for us ------------------
//
// While these were the PartitionNum/EnumerateCap template parameters, a
// nonsensical value was either a compile error or simply unwriteable. As
// runtime configuration they can arrive from a command line, so FanOutSpec
// validates them at construction and composition_fan_out has to cope with
// products that no longer fit in a size_t.

TEST_CASE("FanOutSpec rejects a partition_num below 2",
          "[composition_policy][config]")
{
  // A fan-out of 1 subdivides nothing: every "child" is its own parent, so
  // the search would recurse forever without narrowing a box. Rejected at
  // construction rather than diagnosed as a hang.
  CHECK_THROWS_AS(FanOutSpec {1}, cuminlp::InvalidConfiguration);
  CHECK_THROWS_AS(FanOutSpec {0}, cuminlp::InvalidConfiguration);
  CHECK_NOTHROW(FanOutSpec {2});
}

TEST_CASE("FanOutSpec defaults enumerate_cap to partition_num",
          "[composition_policy][config]")
{
  // Mirrors the old `EnumerateCap = PartitionNum` template default.
  FanOutSpec const spec {7};
  CHECK(spec.partition_num() == 7);
  CHECK(spec.enumerate_cap() == 7);
}

TEST_CASE("composition_fan_out saturates instead of wrapping",
          "[composition_policy][config]")
{
  // 10 continuous slots at partition_num 10^7 is 10^70 children -- far past
  // the size_t ceiling. A wrapped product could land on a small number and
  // size every device buffer far too small; saturating makes the caller's
  // budget check reject it instead (see GraphReplay::build).
  Composition<10> huge {};
  huge.fill(SlotKind::Continuous);
  CHECK(cuminlp::composition_fan_out(huge, FanOutSpec {10000000})
        == std::numeric_limits<std::size_t>::max());

  // A product that does fit is still computed exactly.
  Composition<3> small {};
  small.fill(SlotKind::Continuous);
  CHECK(cuminlp::composition_fan_out(small, FanOutSpec {4}) == 64);

  // Padding slots contribute a factor of 1, which is what lets a shorter
  // assignment share a wider Composition's graph.
  Composition<3> padded = {
      SlotKind::Continuous, SlotKind::Padding, SlotKind::Padding};
  CHECK(cuminlp::composition_fan_out(padded, FanOutSpec {4}) == 4);
}

TEST_CASE("A policy's fan_out is what its callers decode against",
          "[composition_policy][config]")
{
  // The reason FanOutSpec lives on the policy: CompositionInterval::materialise
  // reads it off the same object it calls choose() on, so the widths used to
  // encode a node's sidx and to decode it cannot come apart. Previously the
  // driver and policy carried independent EnumerateCap template arguments.
  GreedyCompositionPolicy<double, 2> policy {FanOutSpec {4, 50}};
  CHECK(policy.fan_out().partition_num() == 4);
  CHECK(policy.fan_out().enumerate_cap() == 50);
}

// --- Capacity vs count -----------------------------------------------------
//
// Capacity is a compile-time array bound (partition::SlotContext is
// register-resident); count is how many slots a node actually fills. The
// separation is what lets one compiled rung serve every narrower request, so
// the properties that make rounding-up safe are worth pinning down.

TEST_CASE("choose reports the number of slots it filled, per box",
          "[composition_policy][capacity]")
{
  // Same policy, same capacity, different boxes -> different counts. This is
  // the per-node variation the whole capacity/count split exists to allow.
  std::vector<VarKind> kinds = {
      VarKind::Binary, VarKind::Binary, VarKind::Binary};
  GreedyCompositionPolicy<double, 8> policy {FanOutSpec {4}};

  std::vector<cu::interval<double>> all_live = {
      {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  CHECK(policy.choose(all_live, kinds).composition.count == 3);

  std::vector<cu::interval<double>> one_resolved = {
      {0.0, 1.0}, {1.0, 1.0}, {0.0, 1.0}};
  CHECK(policy.choose(one_resolved, kinds).composition.count == 2);

  std::vector<cu::interval<double>> all_resolved = {
      {0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}};
  CHECK(policy.choose(all_resolved, kinds).composition.count == 0);
}

TEST_CASE("A wider capacity produces the same children as a narrow one",
          "[composition_policy][capacity]")
{
  // The property that makes the ladder's round-up free: surplus slots are
  // Padding, Padding has fan-out 1, so the region count is unchanged. If this
  // failed, rounding capacity 4 up to rung 8 would silently change the search.
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Binary};
  FanOutSpec const fan_out {4};

  GreedyCompositionPolicy<double, 2> narrow {fan_out};
  GreedyCompositionPolicy<double, 8> wide {fan_out};

  auto const narrow_comp = narrow.choose(box, kinds).composition;
  auto const wide_comp = wide.choose(box, kinds).composition;

  CHECK(narrow_comp.count == wide_comp.count);
  CHECK(cuminlp::composition_fan_out(narrow_comp, fan_out)
        == cuminlp::composition_fan_out(wide_comp, fan_out));
}

TEST_CASE("max_cycle_size caps the slots a policy fills below its capacity",
          "[composition_policy][capacity]")
{
  // Five live variables, capacity 8, but the cap says use 3. Without the cap
  // being honoured in fill_*, --max-cycle-size would do nothing at all.
  std::vector<cu::interval<double>> box = {
      {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  std::vector<VarKind> kinds(5, VarKind::Binary);

  GreedyCompositionPolicy<double, 8> policy {
      FanOutSpec {4}, cuminlp::SearchCalibration {.max_cycle_size = 3}};
  auto const assignment = policy.choose(box, kinds);

  CHECK(assignment.composition.count == 3);
  CHECK(assignment.composition[3] == SlotKind::Padding);

  // And the capped box is NOT fathomable: 5 live variables don't fit in 3
  // slots, even though every filled slot enumerates.
  CHECK(is_fully_enumerable(assignment.composition));
  CHECK_FALSE(can_fathom_without_children(5, assignment.composition));
}

TEST_CASE("possible_compositions respects max_cycle_size",
          "[composition_policy][capacity]")
{
  // The driver sizes its work off this enumeration, so it must not contain
  // compositions choose() would never return.
  std::vector<VarKind> kinds(6, VarKind::Binary);

  GreedyCompositionPolicy<double, 8> policy {
      FanOutSpec {4}, cuminlp::SearchCalibration {.max_cycle_size = 2}};

  for (auto const& comp : policy.possible_compositions(kinds)) {
    CHECK(comp.count <= 2);
  }
}

TEST_CASE("A policy cannot be capped above the capacity it was compiled for",
          "[composition_policy][capacity]")
{
  // Asking for more slots than the instantiated SlotContext holds is a
  // configuration error, not something to silently clamp: the fix is a wider
  // ladder rung.
  CHECK_THROWS_AS((GreedyCompositionPolicy<double, 8> {
                      FanOutSpec {4},
                      cuminlp::SearchCalibration {.max_cycle_size = 16}}),
                  cuminlp::InvalidConfiguration);

  // Unset (0) means "the whole capacity".
  GreedyCompositionPolicy<double, 8> policy {FanOutSpec {4}};
  CHECK(policy.max_cycle_size() == 8);
}
