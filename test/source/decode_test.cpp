// Host-only tests for cuminlp::decode::slot_bounds (slot_decode.hpp) and
// search::CompositionInterval::materialise (search.hpp). Both are pure
// functions of a box/composition/index, with no CUDA dependency despite
// slot_bounds also being the exact function partition.cuh's __device__
// get_slot_bounds calls -- see TEST_EXTENSION.md §4a: host/device agreement
// is structural (one shared function), not something proven by a separate
// device-side test.
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/search.hpp"
#include "cuminlp/slot_decode.hpp"

using cuminlp::Composition;
using cuminlp::FanOutSpec;
using cuminlp::GreedyCompositionPolicy;
using cuminlp::SlotKind;
using cuminlp::dag::VarKind;
using cuminlp::decode::slot_bounds;
using cuminlp::search::CompositionInterval;
using cuminlp::search::IntervalHistory;

namespace
{
// Every value compared here is either passed straight through or computed
// by one deterministic arithmetic step on exactly-representable inputs, so
// bitwise equality is the right check -- this sidesteps -Wfloat-equal
// (which flags == / != between floats, but not < / >) rather than
// weakening the assertion to an approximate one.
bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}
}  // namespace

// §3a: the union of a node's children must contain the parent box -- no
// gaps at the seams or the outer edges. Checked directionally (containment),
// not exact tiling, since lb + width*fan_out != ub in general floating point.
TEST_CASE(
    "Continuous/IntegerBisect children's union contains the parent interval",
    "[decode][3a]")
{
  cu::interval<double> parent {0.0, 10.0};
  constexpr std::size_t fan_out = 4;

  cu::interval<double> children[fan_out];
  for (std::size_t part = 0; part < fan_out; ++part) {
    slot_bounds<double>(
        SlotKind::Continuous, parent, part, fan_out, children[part]);
  }

  // Outer edges.
  CHECK(feq(children[0].lb, parent.lb));
  CHECK(feq(children[fan_out - 1].ub, parent.ub));

  // Seams: consecutive children touch (>= for over-coverage tolerance; a
  // gap, i.e. children[i].ub < children[i+1].lb, must never happen).
  for (std::size_t i = 0; i + 1 < fan_out; ++i) {
    CHECK(children[i].ub >= children[i + 1].lb);
  }

  // Every part strictly inside the parent.
  for (auto& c : children) {
    CHECK(c.lb >= parent.lb);
    CHECK(c.ub <= parent.ub);
  }
}

TEST_CASE(
    "Continuous decode handles a degenerate (lb == ub) parent without "
    "producing a gap",
    "[decode][3a]")
{
  cu::interval<double> parent {5.0, 5.0};
  cu::interval<double> child;
  slot_bounds<double>(SlotKind::Continuous, parent, 0, 4, child);

  CHECK(feq(child.lb, 5.0));
  CHECK(feq(child.ub, 5.0));
}

TEST_CASE("fan_out == 1 reproduces the parent interval exactly", "[decode][3a]")
{
  cu::interval<double> parent {-3.0, 7.0};
  cu::interval<double> child;
  slot_bounds<double>(SlotKind::Continuous, parent, 0, 1, child);

  CHECK(feq(child.lb, parent.lb));
  CHECK(feq(child.ub, parent.ub));
}

// §3b: IntegerEnumerate/BinaryEnumerate snap to the integer lattice even
// when the parent box isn't already aligned (the normal case after an
// IntegerBisect).
TEST_CASE(
    "IntegerEnumerate covers every integer in a lattice-aligned parent domain",
    "[decode][3b]")
{
  cu::interval<double> parent {0.0, 3.0};  // 4 integers: 0,1,2,3
  for (std::size_t part = 0; part < 4; ++part) {
    cu::interval<double> child;
    slot_bounds<double>(SlotKind::IntegerEnumerate, parent, part, 50, child);
    CHECK(feq(child.lb, child.ub));
    CHECK(feq(child.lb, static_cast<double>(part)));
  }
}

TEST_CASE("IntegerEnumerate snaps to the lattice when the parent isn't aligned",
          "[decode][3b]")
{
  // Reachable directly from an IntegerBisect child: parent.lb == 2.4 is not
  // an integer. The enumerate decode must still land on integers: 3, 4, 5.
  cu::interval<double> parent {2.4, 5.6};
  for (std::size_t part = 0; part < 3; ++part) {
    cu::interval<double> child;
    slot_bounds<double>(SlotKind::IntegerEnumerate, parent, part, 50, child);
    CHECK(feq(child.lb, child.ub));
    CHECK(feq(child.lb, 3.0 + static_cast<double>(part)));
  }
}

TEST_CASE(
    "IntegerEnumerate clamps rather than exceeding the parent's upper bound",
    "[decode][3b]")
{
  cu::interval<double> parent {0.0, 3.0};
  cu::interval<double> child;
  // fan_out (50) wider than the true remaining domain (4 integers): part
  // indices beyond the domain must clamp to parent.ub, not read past it.
  slot_bounds<double>(SlotKind::IntegerEnumerate, parent, 10, 50, child);

  CHECK(feq(child.lb, parent.ub));
  CHECK(feq(child.ub, parent.ub));
}

TEST_CASE("BinaryEnumerate decodes 0 and 1 exactly", "[decode][3b]")
{
  cu::interval<double> parent {0.0, 1.0};
  cu::interval<double> c0, c1;
  slot_bounds<double>(SlotKind::BinaryEnumerate, parent, 0, 2, c0);
  slot_bounds<double>(SlotKind::BinaryEnumerate, parent, 1, 2, c1);

  CHECK(feq(c0.lb, 0.0));
  CHECK(feq(c0.ub, 0.0));
  CHECK(feq(c1.lb, 1.0));
  CHECK(feq(c1.ub, 1.0));
}

TEST_CASE("Padding leaves the dimension unchanged", "[decode]")
{
  cu::interval<double> parent {-2.0, 9.0};
  cu::interval<double> child;
  slot_bounds<double>(SlotKind::Padding, parent, 0, 1, child);

  CHECK(feq(child.lb, parent.lb));
  CHECK(feq(child.ub, parent.ub));
}

// §4a: materialise() at pidx == 0 uses the Problem's actual root box, not an
// unbounded-domain fallback disconnected from the real problem.
TEST_CASE(
    "CompositionInterval::materialise at pidx == 0 returns the given root box",
    "[decode][4a]")
{
  constexpr std::size_t CYCLE_SIZE = 1;
  std::vector<VarKind> kinds = {VarKind::Continuous};
  std::vector<cu::interval<double>> root_box = {{-1.0, 1.0}};

  IntervalHistory<double> history;
  GreedyCompositionPolicy<double, CYCLE_SIZE> policy {FanOutSpec {4}};

  CompositionInterval<double, CYCLE_SIZE> node {
      .sidx = 0, .pidx = 0, .depth = 0, .lb = 0.0};

  std::vector<cu::interval<double>> out;
  node.materialise(history, out, policy, kinds, root_box);

  REQUIRE(out.size() == 1);
  CHECK(feq(out[0].lb, -1.0));
  CHECK(feq(out[0].ub, 1.0));
}

TEST_CASE(
    "CompositionInterval::materialise decodes a non-root node by "
    "reconstructing its " "parent's Composition",
    "[decode][4a]")
{
  constexpr std::size_t CYCLE_SIZE = 1;
  std::vector<VarKind> kinds = {VarKind::Continuous};
  std::vector<cu::interval<double>> root_box = {{0.0, 8.0}};

  IntervalHistory<double> history;
  history.enqueue({});  // index 0: unused sentinel, mirrors GraphDriver's usage
  std::size_t parent_idx =
      history.enqueue({{0.0, 8.0}});  // index 1: the real parent box

  GreedyCompositionPolicy<double, CYCLE_SIZE> policy {FanOutSpec {4}};

  // The policy bisects the one continuous variable 4-ways; sidx == 2 is the
  // third quarter: [4.0, 6.0].
  // slot_count must match what the policy fills for this parent box (one
  // continuous variable, so one slot); materialise() checks it before
  // decoding, since a disagreement means sidx would be decoded against the
  // wrong radix vector.
  CompositionInterval<double, CYCLE_SIZE> node {.sidx = 2,
                                               .pidx = parent_idx,
                                               .depth = 1,
                                               .lb = 0.0,
                                               .slot_count = 1};

  std::vector<cu::interval<double>> out;
  node.materialise(history, out, policy, kinds, root_box);

  REQUIRE(out.size() == 1);
  CHECK(feq(out[0].lb, 4.0));
  CHECK(feq(out[0].ub, 6.0));
}

TEST_CASE(
    "materialise rejects a node whose slot_count the policy no longer agrees "
    "with",
    "[decode][4a]")
{
  // The purity tripwire. A policy that consulted state changing during a
  // solve would return a different assignment here than the one that encoded
  // sidx, and every bound decoded below would be against the wrong radix
  // vector -- a wrong box, not an error. Simulated by handing materialise a
  // slot_count that disagrees with what the policy returns for this parent.
  constexpr std::size_t CYCLE_SIZE = 2;
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};
  std::vector<cu::interval<double>> root_box = {{0.0, 8.0}, {0.0, 8.0}};

  IntervalHistory<double> history;
  history.enqueue({});
  std::size_t parent_idx = history.enqueue({{0.0, 8.0}, {0.0, 8.0}});

  GreedyCompositionPolicy<double, CYCLE_SIZE> policy {FanOutSpec {4}};

  // The policy fills 2 slots for this box; claim 1.
  CompositionInterval<double, CYCLE_SIZE> node {.sidx = 2,
                                                .pidx = parent_idx,
                                                .depth = 1,
                                                .lb = 0.0,
                                                .slot_count = 1};

  std::vector<cu::interval<double>> out;
  CHECK_THROWS_AS(
      node.materialise(history, out, policy, kinds, root_box),
      cuminlp::InvalidConfiguration);
}

TEST_CASE(
    "materialise does not let a Padding slot reset a variable an earlier slot "
    "narrowed",
    "[decode][4a]")
{
  // Regression test. `choose` fills a Padding slot's var_id by repeating the
  // last variable it assigned, and decode::slot_bounds for Padding writes
  // out[dim] = parent[dim]. If materialise walks the padding tail, that write
  // reverts the narrowing the real slot performed on the same dimension, and
  // the caller gets the parent's box back for that variable -- a strictly
  // wider box than the node actually represents, so the search would report a
  // bound it never proved.
  //
  // It is specifically a host/device divergence: get_slot_bounds returns on
  // its first var_ids match, so the device always used the real slot.
  //
  // Reachable whenever live variables < slots, which is most of a solve's
  // later nodes; capacity rounding up to a ladder rung makes it the norm.
  constexpr std::size_t CAPACITY = 4;
  std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Continuous};
  std::vector<cu::interval<double>> root_box = {{0.0, 8.0}, {0.0, 8.0}};

  IntervalHistory<double> history;
  history.enqueue({});
  // Variable 1 is already resolved, so only variable 0 is live: the policy
  // fills one slot and pads the other three, each repeating var_id 0.
  std::size_t parent_idx = history.enqueue({{0.0, 8.0}, {5.0, 5.0}});

  GreedyCompositionPolicy<double, CAPACITY> policy {FanOutSpec {4}};
  auto const assignment =
      policy.choose(history.intervals[parent_idx], kinds);
  REQUIRE(assignment.composition.count == 1);
  REQUIRE(assignment.composition[1] == SlotKind::Padding);
  REQUIRE(assignment.var_ids[1] == 0);  // the padding slot repeats var 0

  CompositionInterval<double, CAPACITY> node {.sidx = 2,
                                              .pidx = parent_idx,
                                              .depth = 1,
                                              .lb = 0.0,
                                              .slot_count = 1};

  std::vector<cu::interval<double>> out;
  node.materialise(history, out, policy, kinds, root_box);

  REQUIRE(out.size() == 2);
  // Third quarter of [0, 8]. Not [0, 8], which is what the padding tail
  // would have restored.
  CHECK(feq(out[0].lb, 4.0));
  CHECK(feq(out[0].ub, 6.0));
  CHECK(feq(out[1].lb, 5.0));
  CHECK(feq(out[1].ub, 5.0));
}
