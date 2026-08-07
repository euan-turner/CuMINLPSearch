#include <algorithm>
#include <limits>
#include <vector>

#include "cuminlp/region/composition.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/bisection_budget.hpp"
#include "cuminlp/policy/greedy_enum.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/region/materialise.hpp"

using cuminlp::ShapeMismatch;
using cuminlp::model::VarKind;
using cuminlp::policy::BisectionBudgetCompositionPolicy;
using cuminlp::policy::GreedyEnumCompositionPolicy;
using cuminlp::region::can_fathom_without_children;
using cuminlp::region::Composition;
using cuminlp::region::FanOutSpec;
using cuminlp::region::is_fully_enumerable;
using cuminlp::region::SlotAssignment;
using cuminlp::region::SlotKind;

TEST_CASE("GreedyEnumCompositionPolicy fills binary slots before integer slots",
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
  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::BinaryEnumerate);
  CHECK(assignment.var_ids[1] == 2);
  CHECK(assignment.composition[1] == SlotKind::IntegerEnumerate);
}

TEST_CASE(
    "GreedyEnumCompositionPolicy partitions integer domains above PartitionNum",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.composition[0] == SlotKind::IntegerPartition);
}

TEST_CASE(
    "GreedyEnumCompositionPolicy's enumerate_cap is independent of partition_num's "
    "partition width",
    "[composition_policy]")
{
  // Domain size 40 -- not enumerable under PartitionNum (4) alone, but is
  // under an EnumerateCap (50) that doesn't also force continuous/partitioned
  // slots to a width of 50.
  std::vector<cu::interval<double>> box = {{0.0, 39.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Continuous};

  GreedyEnumCompositionPolicy<double> policy {
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
    "GreedyEnumCompositionPolicy prefers the smallest remaining integer domain",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 100.0}, {0.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Integer};

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {50}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
  CHECK(assignment.composition[0] == SlotKind::IntegerEnumerate);
}

TEST_CASE("GreedyEnumCompositionPolicy prefers the widest continuous variable",
          "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{0.0, 2.0}, {0.0, 10.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 1);
}

TEST_CASE("GreedyEnumCompositionPolicy spills into the next kind once one runs out",
          "[composition_policy]")
{
  // Only one binary available, but two slots to fill; second slot should
  // fall through to the integer.
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 5.0}};
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Integer};

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  CHECK(assignment.var_ids[0] == 0);
  CHECK(assignment.var_ids[1] == 1);
}

TEST_CASE(
    "GreedyEnumCompositionPolicy returns an empty composition once every "
    "variable is resolved",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {{3.0, 3.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous};

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  auto assignment = policy.choose(box, kinds);

  // A composition is exactly its live slots (design/MODULE_REFACTOR.md §4.6):
  // nothing live means nothing to fill, not a padded-out compile-time width.
  CHECK(assignment.composition.size() == 0);
  CHECK(assignment.var_ids.empty());
}

TEST_CASE(
    "GreedyEnumCompositionPolicy::choose is a pure function of (box, var_kinds)",
    "[composition_policy]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0},
      {0.0, 1.0},
      {0.0, 3.0},
  };
  std::vector<VarKind> kinds = {
      VarKind::Continuous, VarKind::Binary, VarKind::Integer};

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  auto first = policy.choose(box, kinds);
  auto second = policy.choose(box, kinds);

  CHECK(first.composition == second.composition);
  CHECK(first.var_ids == second.var_ids);
}

TEST_CASE(
    "GreedyEnumCompositionPolicy::choose throws ShapeMismatch on an empty "
    "var_kinds span",
    "[composition_policy]")
{
  // A zero-variable box has no valid var_id to pad slots with (0 isn't < 0);
  // Problem::validate() now rejects zero-variable Problems too, but the
  // policy's own contract should reject this directly rather than silently
  // writing an out-of-range var_id (TEST_EXTENSION.md).
  std::vector<cu::interval<double>> box;
  std::vector<VarKind> kinds;

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  REQUIRE_THROWS_AS(policy.choose(box, kinds), ShapeMismatch);
}

TEST_CASE(
    "integer_domain_size does not underflow on a sub-box with no integer point",
    "[composition_policy][3b]")
{
  using Policy = GreedyEnumCompositionPolicy<double>;

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

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
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

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
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
  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4, 50}};
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
  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};

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

  GreedyEnumCompositionPolicy<double> policy {
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
  CHECK_THROWS_AS((GreedyEnumCompositionPolicy<double> {
                      FanOutSpec {4},
                      cuminlp::config::SearchCalibration {
                          .max_cycle_size = cuminlp::config::kMaxSlots + 1}}),
                  cuminlp::InvalidConfiguration);

  // Unset (0) means "the full search cap".
  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
  CHECK(policy.max_cycle_size() == cuminlp::config::kMaxSlots);
}

// --- The var_ids contract (§4.5) --------------------------------------------
//
// choose() must return pairwise-distinct var_ids, each indexing a live
// (lb < ub) dimension -- GreedyEnumCompositionPolicy satisfies this by
// construction; assignment_is_distinct_and_live is the debug assertion
// SearchDriver checks it with.

TEST_CASE("GreedyEnumCompositionPolicy always satisfies the distinct-and-live "
          "var_ids contract",
          "[composition_policy][4.5]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0}, {0.0, 1.0}, {0.0, 3.0}, {5.0, 5.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous,
                                VarKind::Binary,
                                VarKind::Integer,
                                VarKind::Continuous};

  GreedyEnumCompositionPolicy<double> policy {FanOutSpec {4}};
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

// =============================================================================
// BisectionBudgetCompositionPolicy (design/BUDGETED_PARTITION.md)
// =============================================================================

namespace
{
// -Werror=float-equal sidestep, same technique as decode_test.cpp/
// host_budget_test.cpp: every value compared below is either passed
// through unchanged or one deterministic arithmetic step apart, so bitwise
// equality is the right check, just spelled through < to avoid the warning.
bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

double width_of(std::size_t vid,
                const SlotAssignment& a,
                const std::vector<cu::interval<double>>& box)
{
  for (std::size_t j = 0; j < a.var_ids.size(); ++j) {
    if (a.var_ids[j] == vid) {
      return (box[vid].ub - box[vid].lb) / static_cast<double>(a.widths[j]);
    }
  }
  return box[vid].ub - box[vid].lb;  // no slot: unchanged
}

std::size_t product_of(const std::vector<std::size_t>& widths)
{
  std::size_t total = 1;
  for (std::size_t w : widths) {
    total *= w;
  }
  return total;
}
}  // namespace

TEST_CASE("BisectionBudgetCompositionPolicy rejects B outside [1, 62]",
          "[bisection_budget]")
{
  CHECK_THROWS_AS((BisectionBudgetCompositionPolicy<double> {0}),
                  cuminlp::InvalidConfiguration);
  CHECK_THROWS_AS((BisectionBudgetCompositionPolicy<double> {63}),
                  cuminlp::InvalidConfiguration);
  CHECK_NOTHROW((BisectionBudgetCompositionPolicy<double> {1}));
  CHECK_NOTHROW((BisectionBudgetCompositionPolicy<double> {62}));
}

TEST_CASE("n_regions is 2^B regardless of composition, except the empty one",
          "[bisection_budget][3]")
{
  BisectionBudgetCompositionPolicy<double> policy {10};
  CHECK(policy.n_regions(Composition {.kinds = {SlotKind::Continuous}}) == 1024);
  CHECK(policy.n_regions(
            Composition {.kinds = {SlotKind::Continuous, SlotKind::Continuous}})
        == 1024);
  CHECK(policy.n_regions(Composition {}) == 1);
}

// §2.1's two named rows.
TEST_CASE("§2.1 row 1: a lopsided box spends the whole budget on the wider "
          "variable",
          "[bisection_budget][7][2.1]")
{
  std::vector<cu::interval<double>> box = {{0.0, 1e6}, {0.0, 1.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};

  BisectionBudgetCompositionPolicy<double> policy {10};
  auto const a = policy.choose(box, kinds);

  // x2 (width 1) never becomes competitive against x1 even after 10
  // halvings (1e6 / 2^10 ~ 976), so it gets no slot at all this cycle.
  REQUIRE(a.composition.size() == 1);
  CHECK(a.var_ids[0] == 0);
  CHECK(a.widths[0] == 1024);
  CHECK(product_of(a.widths) == 1024);
}

TEST_CASE("§2.1 row 2: equal-width variables split the budget evenly",
          "[bisection_budget][7][2.1]")
{
  std::vector<cu::interval<double>> box = {{0.0, 1000.0}, {0.0, 1000.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};

  BisectionBudgetCompositionPolicy<double> policy {10};
  auto const a = policy.choose(box, kinds);

  REQUIRE(a.composition.size() == 2);
  CHECK(a.widths[0] == 32);  // 2^5
  CHECK(a.widths[1] == 32);
  CHECK(product_of(a.widths) == 1024);
}

TEST_CASE("the invariant: product of widths equals n_regions for every "
          "assignment",
          "[bisection_budget][12.4]")
{
  BisectionBudgetCompositionPolicy<double> policy {8};

  struct Case
  {
    std::vector<cu::interval<double>> box;
    std::vector<VarKind> kinds;
  };
  std::vector<Case> const cases = {
      {{{0.0, 5.0}}, {VarKind::Continuous}},
      {{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}},
       {VarKind::Binary, VarKind::Binary, VarKind::Binary}},
      {{{0.0, 100.0}, {0.0, 3.0}, {-5.0, 5.0}},
       {VarKind::Integer, VarKind::Integer, VarKind::Continuous}},
      // A domain-17 integer, not a power of two, exercising §8's rounding.
      {{{0.0, 16.0}}, {VarKind::Integer}},
  };

  for (const Case& c : cases) {
    auto const a = policy.choose(c.box, c.kinds);
    if (a.composition.size() == 0) {
      continue;  // fully resolved; nothing to check
    }
    CHECK(product_of(a.widths) == policy.n_regions(a.composition));
    CHECK(product_of(a.widths) == 256);
  }
}

// §12 item 3: greedy optimality against brute-force enumeration of every
// Σ b_j = B allocation, for a small case.
TEST_CASE("the greedy allocation is optimal for the min-max objective",
          "[bisection_budget][7][optimality]")
{
  std::vector<double> const initial_widths = {100.0, 30.0, 7.0};
  constexpr std::size_t n = 3;
  constexpr std::size_t B = 6;

  // Brute force: every (b0, b1, b2) with b0+b1+b2 == B, minimise the
  // resulting max width.
  double brute_best = std::numeric_limits<double>::max();
  for (std::size_t b0 = 0; b0 <= B; ++b0) {
    for (std::size_t b1 = 0; b0 + b1 <= B; ++b1) {
      std::size_t const b2 = B - b0 - b1;
      double const w0 = initial_widths[0] / static_cast<double>(1ull << b0);
      double const w1 = initial_widths[1] / static_cast<double>(1ull << b1);
      double const w2 = initial_widths[2] / static_cast<double>(1ull << b2);
      brute_best = std::min(brute_best, std::max({w0, w1, w2}));
    }
  }

  std::vector<cu::interval<double>> box(n);
  std::vector<VarKind> kinds(n, VarKind::Continuous);
  for (std::size_t i = 0; i < n; ++i) {
    box[i] = {0.0, initial_widths[i]};
  }

  BisectionBudgetCompositionPolicy<double> policy {B};
  auto const a = policy.choose(box, kinds);

  double greedy_max = 0.0;
  for (std::size_t vid = 0; vid < n; ++vid) {
    greedy_max = std::max(greedy_max, width_of(vid, a, box));
  }
  CHECK(feq(greedy_max, brute_best));
}

TEST_CASE("greedy allocation is invariant to the order variables appear in",
          "[bisection_budget][determinism]")
{
  BisectionBudgetCompositionPolicy<double> policy {8};

  std::vector<cu::interval<double>> box_a = {{0.0, 1000.0}, {0.0, 10.0}};
  std::vector<cu::interval<double>> box_b = {{0.0, 10.0}, {0.0, 1000.0}};
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};

  auto const assignment_a = policy.choose(box_a, kinds);
  auto const assignment_b = policy.choose(box_b, kinds);

  // The wide variable, whichever index it sits at, gets the larger share.
  double const wide_a = width_of(0, assignment_a, box_a);
  double const narrow_a = width_of(1, assignment_a, box_a);
  double const wide_b = width_of(1, assignment_b, box_b);
  double const narrow_b = width_of(0, assignment_b, box_b);

  CHECK(feq(wide_a, wide_b));
  CHECK(feq(narrow_a, narrow_b));
}

// §8.2: when even the cheapest enumerate slot does not fit, the node
// partitions instead.
TEST_CASE("§8.2 fallback: an integer domain too wide for the budget "
          "partitions instead of enumerating",
          "[bisection_budget][8.2]")
{
  // Domain size ~2^20; B = 5 cannot afford ceil(log2(2^20)) = 20 bisections.
  std::vector<cu::interval<double>> box = {{0.0, 1048575.0}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  BisectionBudgetCompositionPolicy<double> policy {5};
  auto const a = policy.choose(box, kinds);

  REQUIRE(a.composition.size() == 1);
  CHECK(a.composition[0] == SlotKind::IntegerPartition);
  CHECK(a.widths[0] == 32);  // 2^5: all of B, since it's the only candidate
}

TEST_CASE("§8.2 fallback: a mix where the cheap integer enumerates and the "
          "wide one partitions",
          "[bisection_budget][8.2]")
{
  std::vector<cu::interval<double>> box = {{0.0, 3.0},  // domain 4: cost 2
                                           {0.0, 1048575.0}};  // domain ~2^20
  std::vector<VarKind> kinds = {VarKind::Integer, VarKind::Integer};

  BisectionBudgetCompositionPolicy<double> policy {5};
  auto const a = policy.choose(box, kinds);

  REQUIRE(a.composition.size() == 2);
  for (std::size_t j = 0; j < a.composition.size(); ++j) {
    if (a.var_ids[j] == 0) {
      CHECK(a.composition[j] == SlotKind::IntegerEnumerate);
      CHECK(a.widths[j] == 4);
      CHECK(a.domain_sizes[j] == 4);
    } else {
      CHECK(a.composition[j] == SlotKind::IntegerPartition);
    }
  }
  CHECK(product_of(a.widths) == 32);
}

// §12 item 5: duplicate suppression. For a rounded-up enumerate slot,
// exactly the children whose digit is >= d_j are flagged, and each flagged
// child materialises to a box identical to its d_j - 1 sibling.
TEST_CASE("duplicate children are flagged and decode identically to their "
          "top sibling",
          "[bisection_budget][8.1]")
{
  // Domain size 5 (0..4) rounds up to width 8 (ceil(log2(5)) == 3).
  std::vector<cu::interval<double>> box = {{0.0, 4.0}};
  std::vector<VarKind> kinds = {VarKind::Integer};

  BisectionBudgetCompositionPolicy<double> policy {3};
  auto const a = policy.choose(box, kinds);
  REQUIRE(a.composition.size() == 1);
  REQUIRE(a.widths[0] == 8);
  REQUIRE(a.domain_sizes[0] == 5);

  std::vector<std::size_t> const prefix = cuminlp::region::slot_prefixes(a.widths);

  std::vector<cu::interval<double>> sibling;
  cuminlp::region::materialise<double>(box, 4, a, sibling);  // r=4: the top real point

  for (std::size_t r = 0; r < 8; ++r) {
    bool const expected_duplicate = r >= 5;
    CHECK(cuminlp::region::is_duplicate_child(r, a, prefix)
          == expected_duplicate);
    if (expected_duplicate) {
      std::vector<cu::interval<double>> child;
      cuminlp::region::materialise<double>(box, r, a, child);
      REQUIRE(child.size() == 1);
      CHECK(feq(child[0].lb, sibling[0].lb));
      CHECK(feq(child[0].ub, sibling[0].ub));
    }
  }
}

// §7.1's boundary: leftover budget with no bisecting pool (all-binary,
// B > live binary count) still spends every bisection somewhere.
TEST_CASE("leftover budget with no bisecting pool pads the widest taken "
          "enumerate slot",
          "[bisection_budget][8.1][edge]")
{
  std::vector<cu::interval<double>> box = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<VarKind> kinds = {VarKind::Binary, VarKind::Binary};

  BisectionBudgetCompositionPolicy<double> policy {5};  // 2 binaries, B=5
  auto const a = policy.choose(box, kinds);

  REQUIRE(a.composition.size() == 2);
  CHECK(product_of(a.widths) == 32);  // == N; the invariant holds exactly
  for (SlotKind k : a.composition) {
    CHECK(k == SlotKind::BinaryEnumerate);
  }
}
