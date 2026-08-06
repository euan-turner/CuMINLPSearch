#include <algorithm>
#include <limits>
#include <vector>

#include "cuminlp/region/composition.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/greedy.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/fan_out.hpp"

using cuminlp::ShapeMismatch;
using cuminlp::model::VarKind;
using cuminlp::policy::GreedyCompositionPolicy;
using cuminlp::region::can_fathom_without_children;
using cuminlp::region::Composition;
using cuminlp::region::FanOutSpec;
using cuminlp::region::is_fully_enumerable;
using cuminlp::region::SlotKind;

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
  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::BinaryEnumerate);
  CHECK(assignment.var_ids[1] == 2);
  CHECK(assignment.composition[1] == SlotKind::IntegerEnumerate);
}

TEST_CASE(
    "GreedyCompositionPolicy partitions integer domains above PartitionNum",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.composition[0] == SlotKind::IntegerPartition);
}

TEST_CASE(
    "GreedyCompositionPolicy's enumerate_cap is independent of partition_num's "
    "partition width",
    "[composition_policy]")
{
  // Domain size 40 -- not enumerable under PartitionNum (4) alone, but is
  // under an EnumerateCap (50) that doesn't also force continuous/partitioned
  // slots to a width of 50.
  std::vector<cu::interval<double>> box = {{0.0, 39.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Continuous};

  GreedyCompositionPolicy<double> policy {
      FanOutSpec {/*partition_num=*/4, /*enumerate_cap=*/50}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
  CHECK(assignment.var_ids[1] == 1);
  CHECK(assignment.composition[1] == SlotKind::Continuous);

  // The continuous slot still partitions at PartitionNum (4), not EnumerateCap.
  CHECK(cuminlp::region::slot_fan_out(SlotKind::Continuous, FanOutSpec {4, 50})
        == 4);
  CHECK(cuminlp::region::slot_fan_out(SlotKind::IntegerEnumerate,
                                      FanOutSpec {4, 50})
        == 50);
}

TEST_CASE(
    "GreedyCompositionPolicy prefers the smallest remaining integer domain",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}, {0.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Integer};

  GreedyCompositionPolicy<double> policy {FanOutSpec {50}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
}

TEST_CASE("GreedyCompositionPolicy prefers the widest continuous variable",
          "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 2.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
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

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.var_ids[1] == 1);
}

TEST_CASE(
    "GreedyCompositionPolicy returns an empty composition once every "
    "variable is resolved",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  // A composition is exactly its live slots (design/MODULE_REFACTOR.md §4.6):
  // nothing live means nothing to fill, not a padded-out compile-time width.
  CHECK(assignment.composition.size() == 0);
  CHECK(assignment.var_ids.empty());
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

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
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

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  REQUIRE_THROWS_AS(policy.choose(box, kinds), ShapeMismatch);
}

TEST_CASE(
    "integer_domain_size does not underflow on a sub-box with no integer point",
    "[composition_policy][3b]")
{
  using Policy = GreedyCompositionPolicy<double>;

  // A normal, lattice-aligned domain still counts correctly.
  CHECK(Policy::integer_domain_size({0.0, 3.0}) == 4);

  // Reachable directly from an IntegerPartition child: ceil(lb) > floor(ub),
  // no integer point at all. Must classify as 0 (empty), not wrap to a huge
  // size_t via an out-of-range double->size_t cast.
  CHECK(Policy::integer_domain_size({2.5, 2.9}) == 0);
  CHECK(Policy::integer_domain_size({2.5, 2.9}) < 1000);
}

TEST_CASE(
    "A sub-box with no integer point is still classified as enumerable rather "
    "than " "partitioned forever",
    "[composition_policy][3b]")
{
  // Under the old integer_domain_size, {2.5, 2.9}'s domain size underflowed
  // to a huge value, comparing as > EnumerateCap and driving another
  // partition forever. With the fix, its (empty) domain size of 0 is <= any
  // EnumerateCap, so it terminates as IntegerEnumerate instead.
  std::vector<cu::interval<double>> box = {{2.5, 2.9}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
}

// TEST_EXTENSION.md: the fathoming predicate, tested directly and
// independent of the surrounding search loop (see search/driver.hpp).
TEST_CASE("can_fathom_without_children at live_count == slot count",
          "[composition_policy][4b]")
{
  Composition comp {.kinds = {SlotKind::BinaryEnumerate,
                              SlotKind::BinaryEnumerate,
                              SlotKind::BinaryEnumerate}};
  CHECK(can_fathom_without_children(3, comp));
}

TEST_CASE(
    "can_fathom_without_children is false when live variables outnumber the "
    "slots the policy filled",
    "[composition_policy][4b]")
{
  // Two enumerable slots, but three live variables: testing against a wider
  // bound than the policy actually filled would call this fathomable and
  // discard a subtree that may hold the optimum.
  Composition comp {
      .kinds = {SlotKind::BinaryEnumerate, SlotKind::BinaryEnumerate}};
  CHECK(is_fully_enumerable(comp));
  CHECK_FALSE(can_fathom_without_children(3, comp));
  CHECK(can_fathom_without_children(2, comp));
}

TEST_CASE("can_fathom_without_children at live_count == 0 (fully resolved box)",
          "[composition_policy][4b]")
{
  // A fully-resolved box must be fathomed, never partitioned -- otherwise it
  // re-enqueues copies of itself indefinitely (TEST_EXTENSION.md).
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(can_fathom_without_children(0, assignment.composition));
}

TEST_CASE(
    "can_fathom_without_children is false when a live Continuous slot remains",
    "[composition_policy][4b]")
{
  Composition comp {.kinds = {SlotKind::BinaryEnumerate, SlotKind::Continuous}};
  CHECK_FALSE(can_fathom_without_children(2, comp));
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
  Composition huge {.kinds = std::vector<SlotKind>(10, SlotKind::Continuous)};
  CHECK(cuminlp::region::composition_fan_out(huge, FanOutSpec {10000000})
        == std::numeric_limits<std::size_t>::max());

  // A product that does fit is still computed exactly.
  Composition small {.kinds = std::vector<SlotKind>(3, SlotKind::Continuous)};
  CHECK(cuminlp::region::composition_fan_out(small, FanOutSpec {4}) == 64);
}

TEST_CASE("slot_prefixes agrees with repeated-division digits",
          "[composition_policy][config]")
{
  // (r / prefix[j]) % fan_out[j] must reproduce the same digit a per-thread
  // repeated-division loop would compute for every region r
  // (design/MODULE_REFACTOR.md §4.2) -- the property the two-pass device
  // kernels depend on for bit-identical results against the old scan.
  Composition comp {.kinds = {SlotKind::BinaryEnumerate,
                              SlotKind::IntegerEnumerate,
                              SlotKind::Continuous}};
  FanOutSpec const fan_out {4, 3};  // widths: 2, 3, 4
  auto const prefix = cuminlp::region::slot_prefixes(comp, fan_out);
  REQUIRE(prefix.size() == 3);

  std::size_t const n_regions =
      cuminlp::region::composition_fan_out(comp, fan_out);
  for (std::size_t r = 0; r < n_regions; ++r) {
    std::size_t idx = r;
    for (std::size_t j = 0; j < comp.size(); ++j) {
      std::size_t const width = cuminlp::region::slot_fan_out(comp[j], fan_out);
      std::size_t const expected_digit = idx % width;
      idx /= width;
      CHECK((r / prefix[j]) % width == expected_digit);
    }
  }
}

TEST_CASE("A policy's fan_out is what its callers decode against",
          "[composition_policy][config]")
{
  // The reason FanOutSpec lives on the policy: Node::materialise
  // reads it off the same object it calls choose() on, so the widths used to
  // encode a node's sidx and to decode it cannot come apart.
  GreedyCompositionPolicy<double> policy {FanOutSpec {4, 50}};
  CHECK(policy.fan_out().partition_num() == 4);
  CHECK(policy.fan_out().enumerate_cap() == 50);
}

// --- max_cycle_size / kMaxSlots ---------------------------------------------

TEST_CASE("choose reports the number of slots it filled, per box",
          "[composition_policy][capacity]")
{
  // Same policy, different boxes -> different slot counts.
  std::vector<VarKind> kinds = {
      VarKind::Binary, VarKind::Binary, VarKind::Binary};
  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};

  std::vector<cu::interval<double>> all_live = {
      {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  CHECK(policy.choose(all_live, kinds).composition.size() == 3);

  std::vector<cu::interval<double>> one_resolved = {
      {0.0, 1.0}, {1.0, 1.0}, {0.0, 1.0}};
  CHECK(policy.choose(one_resolved, kinds).composition.size() == 2);

  std::vector<cu::interval<double>> all_resolved = {
      {0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}};
  CHECK(policy.choose(all_resolved, kinds).composition.size() == 0);
}

TEST_CASE("max_cycle_size caps the slots a policy fills",
          "[composition_policy][capacity]")
{
  // Five live variables, but the cap says use 3. Without the cap being
  // honoured in fill_*, --max-cycle-size would do nothing at all.
  std::vector<cu::interval<double>> box = {
      {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  std::vector<VarKind> kinds(5, VarKind::Binary);

  GreedyCompositionPolicy<double> policy {
      FanOutSpec {4}, cuminlp::config::SearchCalibration {.max_cycle_size = 3}};
  auto const assignment = policy.choose(box, kinds);

  CHECK(assignment.composition.size() == 3);

  // And the capped box is NOT fathomable: 5 live variables don't fit in 3
  // slots, even though every filled slot enumerates.
  CHECK(is_fully_enumerable(assignment.composition));
  CHECK_FALSE(can_fathom_without_children(5, assignment.composition));
}

TEST_CASE("A policy cannot be capped above kMaxSlots",
          "[composition_policy][capacity]")
{
  // Asking for more slots than the search cap allows is a configuration
  // error, not something to silently clamp.
  CHECK_THROWS_AS((GreedyCompositionPolicy<double> {
                      FanOutSpec {4},
                      cuminlp::config::SearchCalibration {
                          .max_cycle_size = cuminlp::config::kMaxSlots + 1}}),
                  cuminlp::InvalidConfiguration);

  // Unset (0) means "the full search cap".
  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  CHECK(policy.max_cycle_size() == cuminlp::config::kMaxSlots);
}

// --- The var_ids contract (§4.5) --------------------------------------------
//
// choose() must return pairwise-distinct var_ids, each indexing a live
// (lb < ub) dimension -- GreedyCompositionPolicy satisfies this by
// construction; assignment_is_distinct_and_live is the debug assertion
// SearchDriver checks it with.

TEST_CASE("GreedyCompositionPolicy always satisfies the distinct-and-live "
          "var_ids contract",
          "[composition_policy][4.5]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0}, {0.0, 1.0}, {0.0, 3.0}, {5.0, 5.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous,
                                VarKind::Binary,
                                VarKind::Integer,
                                VarKind::Continuous};

  GreedyCompositionPolicy<double> policy {FanOutSpec {4}};
  auto const assignment = policy.choose(box, kinds);

  CHECK(cuminlp::policy::assignment_is_distinct_and_live<double>(assignment,
                                                                 box));
}

TEST_CASE("assignment_is_distinct_and_live rejects a duplicate var_id",
          "[composition_policy][4.5]")
{
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 1.0}};
  cuminlp::region::SlotAssignment bad {
      .composition = {.kinds = {SlotKind::BinaryEnumerate,
                                SlotKind::BinaryEnumerate}},
      .var_ids = {0, 0}};

  CHECK_FALSE(
      cuminlp::policy::assignment_is_distinct_and_live<double>(bad, box));
}

TEST_CASE(
    "assignment_is_distinct_and_live rejects a resolved (non-live) " "var_id",
    "[composition_policy][4.5]")
{
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {5.0, 5.0}};
  cuminlp::region::SlotAssignment bad {
      .composition = {.kinds = {SlotKind::BinaryEnumerate,
                                SlotKind::BinaryEnumerate}},
      .var_ids = {0, 1}};

  CHECK_FALSE(
      cuminlp::policy::assignment_is_distinct_and_live<double>(bad, box));
}
