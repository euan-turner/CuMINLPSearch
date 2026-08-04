// Host-only tests for the bounded frontier (design/BOUNDED_FRONTIER.md §8):
// IntervalPQueue::compact and its eviction order, IntervalHistory's refcounted
// GC, the dropped-floor gating of the epilogue's claims, and the budget
// arithmetic that sizes a compaction.
//
// None of it needs a GPU. search.hpp deliberately includes only
// cuinterval/interval.h, and host_budget.hpp is arithmetic on counts and
// bounds -- the CUDA-side wiring (graph_driver.cuh) is the only part that
// cannot be reached from a plain host translation unit. See TEST_EXTENSION.md.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "cuminlp/host_budget.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/search.hpp"

using cuminlp::compact_keep_count;
using cuminlp::DropAccounting;
using cuminlp::FinalBounds;
using cuminlp::finalise_bounds;
using cuminlp::StopReason;
using cuminlp::search::CompositionInterval;
using cuminlp::search::IntervalHistory;
using cuminlp::search::IntervalPQueue;

namespace
{
// Every bound compared here is a literal that survives a copy unchanged, so
// bitwise equality is the right check -- written through < to sidestep
// -Wfloat-equal rather than weakening the assertion (as decode_test.cpp does).
bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

constexpr std::size_t kCapacity = 4;
using Node = CompositionInterval<double, kCapacity>;
using Queue = IntervalPQueue<double, Node>;

Node node(double lb, std::size_t depth = 1, std::size_t pidx = 1)
{
  return Node {
      .sidx = 0, .pidx = pidx, .depth = depth, .lb = lb, .slot_count = 1};
}

// The frontier's elements in priority order, which is the only order the
// driver ever reads them in.
std::vector<double> drain(Queue& q)
{
  std::vector<double> out;
  while (!q.empty()) {
    out.push_back(q.dequeue().lb);
  }
  return out;
}
}  // namespace

// ---------------------------------------------------------------- §8.1

TEST_CASE("compact keeps exactly the n_keep best elements", "[frontier][8.1]")
{
  Queue q(1);
  std::vector<double> const bounds {
      5.0, -2.0, 11.0, 0.5, 7.25, -9.0, 3.0, 100.0, -0.25, 42.0};
  for (double lb : bounds) {
    q.enqueue(node(lb));
  }
  REQUIRE(q.size() == bounds.size());

  constexpr std::size_t n_keep = 4;
  std::vector<double> sorted = bounds;
  std::sort(sorted.begin(), sorted.end());

  std::vector<double> evicted;
  q.compact(n_keep, [&](const Node& e) { evicted.push_back(e.lb); });

  REQUIRE(q.size() == n_keep);
  CHECK(q.capacity() <= 2 * n_keep);

  // Survivors are the n_keep best, and they come out in priority order: the
  // heap invariant survived the rebuild.
  std::vector<double> const survivors = drain(q);
  REQUIRE(survivors.size() == n_keep);
  for (std::size_t i = 0; i < n_keep; ++i) {
    CHECK(feq(survivors[i], sorted[i]));
  }

  // on_evict saw exactly the complement, once each.
  std::sort(evicted.begin(), evicted.end());
  REQUIRE(evicted.size() == bounds.size() - n_keep);
  for (std::size_t i = 0; i < evicted.size(); ++i) {
    CHECK(feq(evicted[i], sorted[n_keep + i]));
  }
}

TEST_CASE("compact preserves count_viable for surviving regions",
          "[frontier][8.1]")
{
  Queue q(1);
  for (double lb : {1.0, 2.0, 3.0, 50.0, 60.0, 70.0}) {
    q.enqueue(node(lb));
  }
  // A gub that keeps the three survivors viable and nothing else.
  constexpr double gub = 10.0;
  REQUIRE(q.count_viable(gub) == 3);

  q.compact(3, [](const Node&) {});

  CHECK(q.size() == 3);
  CHECK(q.count_viable(gub) == 3);
}

TEST_CASE("compact is a no-op when the frontier already fits",
          "[frontier][8.1]")
{
  Queue q(1);
  for (double lb : {1.0, 2.0, 3.0}) {
    q.enqueue(node(lb));
  }

  std::size_t evictions = 0;
  q.compact(3, [&](const Node&) { ++evictions; });
  CHECK(evictions == 0);
  CHECK(q.size() == 3);

  q.compact(99, [&](const Node&) { ++evictions; });
  CHECK(evictions == 0);
  CHECK(q.size() == 3);
  CHECK(drain(q).size() == 3);
}

TEST_CASE("compact to a single element keeps the frontier minimum",
          "[frontier][8.1]")
{
  Queue q(1);
  for (double lb : {4.0, -1.5, 9.0, 2.0}) {
    q.enqueue(node(lb));
  }

  q.compact(1, [](const Node&) {});
  REQUIRE(q.size() == 1);
  CHECK(feq(q.peek().lb, -1.5));
}

// ---------------------------------------------------------------- §8.2

TEST_CASE("eviction takes dominated regions before viable ones",
          "[frontier][8.2]")
{
  constexpr double gub = 10.0;

  Queue q(1);
  // Three viable (lb <= gub) and three dominated, interleaved on the way in so
  // insertion order cannot be what decides.
  for (double lb : {12.0, 1.0, 30.0, 5.0, 11.5, 9.0}) {
    q.enqueue(node(lb));
  }

  DropAccounting dropped;
  q.compact(3, [&](const Node& e) { dropped.fold(e.lb, gub); });

  // Every eviction was of a region the search had already dominated, so this
  // compaction cost the reported bound nothing at all.
  CHECK(dropped.dominated == 3);
  CHECK(dropped.viable == 0);
  CHECK(dropped.lost_nothing());
  CHECK(std::isinf(dropped.lb_min));
}

TEST_CASE("among equal bounds the shallower region is evicted first",
          "[frontier][8.2]")
{
  Queue q(1);
  q.enqueue(node(1.0, /*depth=*/7));
  q.enqueue(node(1.0, /*depth=*/2));
  q.enqueue(node(1.0, /*depth=*/9));

  std::vector<std::size_t> evicted_depths;
  q.compact(2, [&](const Node& e) { evicted_depths.push_back(e.depth); });

  REQUIRE(evicted_depths.size() == 1);
  CHECK(evicted_depths[0] == 2);
}

TEST_CASE("a viable eviction lowers the floor to its bound", "[frontier][8.2]")
{
  constexpr double gub = 10.0;

  Queue q(1);
  for (double lb : {2.0, 3.0, 4.0, 5.0}) {
    q.enqueue(node(lb));
  }

  DropAccounting dropped;
  q.compact(2, [&](const Node& e) { dropped.fold(e.lb, gub); });

  // The two worst survive nothing: both were still viable, so the floor drops
  // to the better of them.
  CHECK(dropped.viable == 2);
  CHECK(dropped.dominated == 0);
  CHECK(feq(dropped.lb_min, 4.0));
  CHECK_FALSE(dropped.lost_nothing());
}

// ---------------------------------------------------------------- §8.3

TEST_CASE("history frees a box once its last child is released",
          "[history][8.3]")
{
  IntervalHistory<double> history;
  REQUIRE(history.enqueue({}) == 0);  // the root sentinel
  std::size_t const baseline = history.live_bytes();

  std::size_t const idx = history.enqueue({{0.0, 1.0}, {2.0, 3.0}});
  CHECK(idx == 1);
  CHECK(history.live_bytes() > baseline);
  CHECK(history.live_count() == 2);

  // Two children name it, then the driver drops its construction reference.
  history.add_ref(idx);
  history.add_ref(idx);
  history.release(idx);
  CHECK(history.ref_count(idx) == 2);
  CHECK(history.live_bytes() > baseline);

  history.release(idx);
  CHECK(history.live_bytes() > baseline);  // one child still holds it

  history.release(idx);
  CHECK(history.live_bytes() == baseline);
  CHECK(history.live_count() == 1);
}

TEST_CASE("history reuses a freed slot", "[history][8.3]")
{
  IntervalHistory<double> history;
  history.enqueue({});

  std::size_t const first = history.enqueue({{0.0, 1.0}});
  history.release(first);  // no children were enqueued: garbage immediately

  std::size_t const second = history.enqueue({{4.0, 8.0}});
  CHECK(second == first);
  CHECK(history.intervals.size() == 2);  // reused, not appended
  CHECK(feq(history.intervals[second][0].lb, 4.0));
}

TEST_CASE("slot 0 is never freed and never reused", "[history][8.3]")
{
  IntervalHistory<double> history;
  REQUIRE(history.enqueue({}) == 0);

  history.release(0);
  history.release(0);
  history.add_ref(0);

  std::size_t const next = history.enqueue({{0.0, 1.0}});
  CHECK(next == 1);
  CHECK(history.intervals.size() == 2);
}

TEST_CASE("materialise resolves a child through a reused slot",
          "[history][8.3]")
{
  // The decode path itself is decode_test.cpp's subject; what matters here is
  // that a pidx pointing at a *recycled* slot reads the box that slot holds
  // now, not the one it held when the index was first handed out.
  IntervalHistory<double> history;
  history.enqueue({});

  std::size_t const stale = history.enqueue({{0.0, 100.0}});
  history.release(stale);

  std::size_t const fresh = history.enqueue({{5.0, 6.0}});
  REQUIRE(fresh == stale);
  CHECK(feq(history.intervals[fresh][0].lb, 5.0));
  CHECK(feq(history.intervals[fresh][0].ub, 6.0));
}

TEST_CASE("live_bytes returns to baseline over an expand/consume cycle",
          "[history][8.3]")
{
  IntervalHistory<double> history;
  history.enqueue({});
  std::size_t const baseline = history.live_bytes();

  for (int round = 0; round < 5; ++round) {
    std::size_t const parent =
        history.enqueue({{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}});
    for (int child = 0; child < 4; ++child) {
      history.add_ref(parent);
    }
    history.release(parent);  // construction reference
    for (int child = 0; child < 4; ++child) {
      history.release(parent);  // each child dequeued or evicted
    }
    CHECK(history.live_bytes() == baseline);
  }
  // Five expansions, and the history never grew past the sentinel plus one.
  CHECK(history.intervals.size() == 2);
}

// ---------------------------------------------------------------- §8.4

namespace
{
constexpr double kNoIncumbent = std::numeric_limits<double>::max();
constexpr double kTolerance = 1e-6;

FinalBounds finalise(double glb,
                     double gub,
                     bool converged,
                     bool pending_empty,
                     double frontier_min,
                     std::size_t viable,
                     const DropAccounting& dropped)
{
  return finalise_bounds(glb,
                         gub,
                         kTolerance,
                         gub < kNoIncumbent,
                         converged,
                         pending_empty,
                         frontier_min,
                         viable,
                         dropped);
}
}  // namespace

TEST_CASE("with nothing dropped the epilogue behaves exactly as before",
          "[epilogue][8.4]")
{
  DropAccounting const none;

  SECTION("an emptied frontier with an incumbent is proven optimal")
  {
    FinalBounds const out = finalise(1.0, 5.0, false, true, 0.0, 0, none);
    CHECK(out.proven_optimal);
    CHECK_FALSE(out.infeasible);
    CHECK(feq(out.glb, 5.0));
  }

  SECTION("an emptied frontier with no incumbent is infeasible")
  {
    FinalBounds const out =
        finalise(1.0, kNoIncumbent, false, true, 0.0, 0, none);
    CHECK_FALSE(out.proven_optimal);
    CHECK(out.infeasible);
    CHECK(feq(out.glb, 1.0));  // left where the loop had it
  }

  SECTION("a stranded frontier is clamped to the incumbent")
  {
    FinalBounds const out = finalise(1.0, 5.0, false, false, 9.0, 0, none);
    CHECK(out.proven_optimal);  // nothing viable left
    CHECK(feq(out.glb, 5.0));
  }

  SECTION("an unfinished run reports the frontier minimum")
  {
    FinalBounds const out = finalise(1.0, 5.0, false, false, 2.0, 3, none);
    CHECK_FALSE(out.proven_optimal);
    CHECK_FALSE(out.infeasible);
    CHECK(feq(out.glb, 2.0));
  }
}

TEST_CASE("a dominated-only drop still permits proven optimality",
          "[epilogue][8.4]")
{
  DropAccounting dropped;
  dropped.fold(/*lb=*/40.0, /*gub=*/5.0);
  REQUIRE(dropped.dominated == 1);
  REQUIRE(dropped.lost_nothing());

  FinalBounds const out = finalise(1.0, 5.0, false, true, 0.0, 0, dropped);
  CHECK(out.proven_optimal);
  CHECK(feq(out.glb, 5.0));
}

TEST_CASE("a viable drop caps the reported lower bound", "[epilogue][8.4]")
{
  DropAccounting dropped;
  dropped.fold(/*lb=*/2.0, /*gub=*/5.0);
  REQUIRE(dropped.viable == 1);

  SECTION("the floor beats an emptied frontier")
  {
    FinalBounds const out = finalise(1.0, 5.0, false, true, 0.0, 0, dropped);
    CHECK_FALSE(out.proven_optimal);  // the gap to the floor is not closed
    CHECK_FALSE(out.infeasible);
    CHECK(feq(out.glb, 2.0));
  }

  SECTION("the floor beats a higher frontier minimum")
  {
    FinalBounds const out = finalise(1.0, 5.0, false, false, 4.0, 1, dropped);
    CHECK(feq(out.glb, 2.0));
  }

  SECTION("a lower frontier minimum still wins")
  {
    FinalBounds const out = finalise(1.0, 5.0, false, false, 1.5, 1, dropped);
    CHECK(feq(out.glb, 1.5));
  }
}

TEST_CASE("a viable drop within tolerance is still proven optimal",
          "[epilogue][8.4]")
{
  DropAccounting dropped;
  dropped.fold(/*lb=*/5.0 - 1e-9, /*gub=*/5.0);
  REQUIRE(dropped.viable == 1);

  FinalBounds const out = finalise(1.0, 5.0, true, false, 5.0, 1, dropped);
  CHECK(out.proven_optimal);
  CHECK(feq(out.glb, 5.0));
}

TEST_CASE("an emptied frontier with viable drops is not infeasible",
          "[epilogue][8.4]")
{
  DropAccounting dropped;
  dropped.fold(/*lb=*/-3.0, /*gub=*/kNoIncumbent);
  REQUIRE(dropped.viable == 1);

  FinalBounds const out =
      finalise(1.0, kNoIncumbent, false, true, 0.0, 0, dropped);
  CHECK_FALSE(out.infeasible);
  CHECK_FALSE(out.proven_optimal);
  // The only honest bracket left: [-3, +inf).
  CHECK(feq(out.glb, -3.0));
}

// ---------------------------------------------------------------- §8.5

TEST_CASE("n_keep is half the nodes the budget can hold", "[budget][8.5]")
{
  // 1000 bytes of room for 10-byte nodes is 100 nodes, so the frontier is
  // halved to 50 and then has room to double back into the budget.
  CHECK(compact_keep_count(2000, 1000, 10) == 50);
  CHECK(compact_keep_count(2000, 0, 10) == 100);
}

TEST_CASE("n_keep floors at one when the history crowds out the frontier",
          "[budget][8.5]")
{
  CHECK(compact_keep_count(1000, 1000, 40) == 1);
  CHECK(compact_keep_count(1000, 999999, 40) == 1);  // history alone is over
  CHECK(compact_keep_count(0, 0, 40) == 1);
}

TEST_CASE("a budget too small for one fan-out stops rather than thrashes",
          "[budget][8.5]")
{
  // The floor of 1 means compact() always has something to keep, so the
  // still-over-budget test after it -- not an empty queue -- is what ends the
  // run. Reproduced here: one node's worth of frontier plus this history is
  // still over budget, so the driver would stop.
  constexpr std::size_t budget = 100;
  constexpr std::size_t history_bytes = 96;
  constexpr std::size_t node_bytes = 40;
  std::size_t const n_keep =
      compact_keep_count(budget, history_bytes, node_bytes);
  CHECK(n_keep == 1);
  CHECK(2 * n_keep * node_bytes + history_bytes > budget);
}

TEST_CASE("frontier_bytes counts capacity rather than size", "[budget][8.5]")
{
  Queue q(64);
  CHECK(q.size() == 0);
  CHECK(q.capacity_bytes() >= 64 * sizeof(Node));

  q.enqueue(node(1.0));
  CHECK(q.size() == 1);
  // Still the whole allocation: what the budget must account for is what was
  // allocated, not what is occupied.
  CHECK(q.capacity_bytes() >= 64 * sizeof(Node));
}

TEST_CASE("a compaction gives the capacity back", "[budget][8.5]")
{
  Queue q(1);
  for (std::size_t i = 0; i < 500; ++i) {
    q.enqueue(node(static_cast<double>(i)));
  }
  std::size_t const before = q.capacity_bytes();

  q.compact(10, [](const Node&) {});

  CHECK(q.capacity_bytes() < before);
  CHECK(q.capacity_bytes() <= 20 * sizeof(Node));
}

// ---------------------------------------------------------------- reporting

TEST_CASE("every stop reason has the spelling the log contract names",
          "[budget][reporting]")
{
  CHECK(std::string(to_string(StopReason::Converged)) == "converged");
  CHECK(std::string(to_string(StopReason::Exhausted)) == "exhausted");
  CHECK(std::string(to_string(StopReason::IterationLimit))
        == "iteration-limit");
  CHECK(std::string(to_string(StopReason::HostMemory)) == "host-memory");
  CHECK(std::string(to_string(StopReason::AllocationFailure))
        == "allocation-failure");
  CHECK(std::string(to_string(StopReason::DeviceMemory)) == "device-memory");
}
