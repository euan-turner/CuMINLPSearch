// design/AGGREGATE_BOUNDING.md §11, tests 3 and 4 -- stage 3's exit criteria.
//
// Host-only: aggregate/selection.hpp and aggregate/frontier.hpp name no CUDA
// type, and this is where the subtle bugs live. The load-bearing test is the
// first one: if `least-lower-bound` does not reproduce search::IntervalPQueue's
// pop order exactly, the aggregate frontier is a *different search* and every
// backend-to-backend comparison stage 6 records is confounded by it.

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cuminlp/aggregate/frontier.hpp"
#include "cuminlp/aggregate/selection.hpp"
#include "cuminlp/search/frontier.hpp"

using cuminlp::aggregate::AggregateFrontier;
using cuminlp::aggregate::AggregateNode;
using cuminlp::aggregate::rebuilds_on_incumbent;
using cuminlp::aggregate::selection_key;
using cuminlp::aggregate::SelectionRule;

namespace
{

constexpr double kNoIncumbent = std::numeric_limits<double>::max();

bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

/// A node's identity for comparison purposes. `search::Node` and
/// `AggregateNode` are different types with the same ordering fields, so the
/// equivalence test compares this rather than either struct.
struct Ident
{
  double lb;
  std::size_t depth;
  std::size_t pidx;
  std::size_t sidx;

  bool operator==(const Ident&) const = default;
};

}  // namespace

TEST_CASE("least-lower-bound reproduces IntervalPQueue's pop order exactly",
          "[aggregate][frontier]")
{
  // Stage 3's exit criterion. Interleaved enqueues and pops rather than
  // fill-then-drain, because the orders can only diverge on a heap that has
  // been sifted in both directions.
  //
  // The generated (lb, depth, pidx) triples are distinct by construction.
  // With a genuine three-way tie both containers are free to return either
  // node -- the comparator says they are interchangeable -- so which one
  // surfaces is a property of the heap's internals, not of the ordering, and
  // asserting on it would test std::push_heap rather than this code.
  std::mt19937 rng {20260811};
  std::uniform_int_distribution<int> action(0, 2);  // 2/3 enqueue, 1/3 pop
  std::uniform_int_distribution<std::size_t> depth_dist(0, 12);
  std::uniform_int_distribution<std::size_t> sidx_dist(0, 3);

  std::set<std::tuple<double, std::size_t, std::size_t>> used;
  std::size_t compared = 0;
  std::size_t next_pidx = 1;

  for (int seq = 0; seq < 60; ++seq) {
    cuminlp::search::IntervalPQueue<double> reference(16);
    AggregateFrontier<double> subject(SelectionRule::LeastLowerBound);
    std::size_t live = 0;
    used.clear();

    for (int step = 0; step < 3200; ++step) {
      if (action(rng) != 0 || live == 0) {
        // Distinct triples: retry until this (lb, depth, pidx) is unused.
        double lb = 0.0;
        std::size_t depth = 0;
        std::size_t pidx = 0;
        do {
          lb = std::round(std::normal_distribution<double>(0.0, 50.0)(rng)
                          * 100.0)
              / 100.0;
          depth = depth_dist(rng);
          pidx = next_pidx++;
        } while (!used.insert({lb, depth, pidx}).second);

        std::size_t const sidx = sidx_dist(rng);
        reference.enqueue(cuminlp::search::Node<double> {
            .sidx = sidx,
            .pidx = pidx,
            .depth = depth,
            .lb = lb,
            .assignment_hash = 0,
        });
        subject.enqueue(AggregateNode<double> {.sidx = sidx,
                                               .pidx = pidx,
                                               .depth = depth,
                                               .lb = lb,
                                               .hull_ub = lb + 1.0,
                                               .branch_hash = 0},
                        kNoIncumbent);
        ++live;
        continue;
      }

      auto const r = reference.dequeue();
      auto const s = subject.pop();
      REQUIRE(Ident {r.lb, r.depth, r.pidx, r.sidx}
              == Ident {s.lb, s.depth, s.pidx, s.sidx});
      --live;
      ++compared;

      // The dual bound the two would report at this moment must also agree.
      REQUIRE(subject.size() == reference.size());
      if (!reference.empty()) {
        REQUIRE(feq(subject.min_lb(), reference.peek().lb));
      }
    }

    while (live > 0) {
      auto const r = reference.dequeue();
      auto const s = subject.pop();
      REQUIRE(Ident {r.lb, r.depth, r.pidx, r.sidx}
              == Ident {s.lb, s.depth, s.pidx, s.sidx});
      --live;
      ++compared;
    }
    REQUIRE(subject.empty());
  }

  // The criterion is >= 1e5 compared pops; assert it rather than trust the
  // loop bounds to keep saying so.
  CHECK(compared >= 100000);
}

TEST_CASE("min_lb is monotone non-decreasing under both rules",
          "[aggregate][frontier]")
{
  // §4.4: with the driver's clamp -- lb(child) := max(aggregate, lb(parent))
  // -- every node's replacements have lb >= its own, so the frontier's
  // lb-minimum only ever rises, regardless of pop order. That is exactly what
  // lets the selection rule change without touching the dual bound.
  //
  // The children below are generated with lb >= the parent's, which is the
  // clamped world this models. Monotonicity is *not* inherent: stage 4 showed
  // an unclamped aggregate over a re-partitioned child can be weaker than the
  // child's own bound. See aggregate_backend_test.cu.
  for (SelectionRule rule : {SelectionRule::LeastLowerBound,
                             SelectionRule::RejectIndex,
                             SelectionRule::RejectIndexStale})
  {
    std::mt19937 rng {99};
    AggregateFrontier<double> frontier(rule);
    double f_best = kNoIncumbent;

    frontier.enqueue(
        AggregateNode<double> {
            .sidx = 0, .pidx = 0, .depth = 0, .lb = -1000.0, .hull_ub = 1000.0},
        f_best);

    double previous = frontier.min_lb();
    std::size_t pidx = 1;

    for (int step = 0; step < 4000 && !frontier.empty(); ++step) {
      auto const parent = frontier.pop();

      // Four children, each with lb >= the parent's -- the only property the
      // monotonicity argument rests on.
      for (std::size_t c = 0; c < 4; ++c) {
        double const lb =
            parent.lb + std::uniform_real_distribution<double>(0.0, 5.0)(rng);
        double const ub =
            lb + std::uniform_real_distribution<double>(0.1, 50.0)(rng);
        if (lb > f_best) {
          continue;  // the driver's cutoff
        }
        frontier.enqueue(AggregateNode<double> {.sidx = c,
                                                .pidx = pidx,
                                                .depth = parent.depth + 1,
                                                .lb = lb,
                                                .hull_ub = ub},
                         f_best);
      }
      ++pidx;

      // An incumbent that improves now and then, so the RejectIndex rebuild
      // path is actually exercised inside the monotonicity check.
      if (step % 50 == 0) {
        double const candidate = parent.lb + 20.0;
        if (candidate < f_best) {
          f_best = candidate;
          frontier.on_incumbent(f_best);
        }
      }

      if (frontier.empty()) {
        break;
      }
      double const now = frontier.min_lb();
      REQUIRE(now >= previous);
      previous = now;
    }
  }
}

TEST_CASE("Lazy deletion never returns a node twice or skips a live one",
          "[aggregate][frontier]")
{
  // The frontier keeps two references per node and discards at most one of
  // them per pop, so every pop leaves a stale reference in the other index.
  // Draining must still yield exactly the enqueued multiset, once each.
  std::mt19937 rng {4242};
  AggregateFrontier<double> frontier(SelectionRule::RejectIndex);
  std::multiset<double> enqueued;
  double f_best = 100.0;
  std::size_t pidx = 1;

  for (int step = 0; step < 20000; ++step) {
    if (std::uniform_int_distribution<int>(0, 2)(rng) != 0 || frontier.empty())
    {
      double const lb =
          std::uniform_real_distribution<double>(-50.0, 50.0)(rng);
      double const ub =
          lb + std::uniform_real_distribution<double>(0.0, 20.0)(rng);
      frontier.enqueue(
          AggregateNode<double> {
              .sidx = 0, .pidx = pidx++, .depth = 1, .lb = lb, .hull_ub = ub},
          f_best);
      enqueued.insert(lb);
    } else {
      auto const n = frontier.pop();
      auto const it = enqueued.find(n.lb);
      REQUIRE(it != enqueued.end());  // never a node that was never enqueued
      enqueued.erase(it);  // ...nor one already returned
    }
    if (step % 1000 == 0) {
      f_best -= 1.0;
      frontier.on_incumbent(f_best);
    }
    REQUIRE(frontier.size() == enqueued.size());
  }

  while (!frontier.empty()) {
    auto const n = frontier.pop();
    auto const it = enqueued.find(n.lb);
    REQUIRE(it != enqueued.end());
    enqueued.erase(it);
  }
  CHECK(enqueued.empty());

  // Garbage collection has to actually run, or the churn above proved nothing
  // about it: 20k steps against a bound that stays proportional to the live
  // set is only possible if stale references are being reclaimed.
  CHECK(frontier.capacity_bytes() < 20000 * sizeof(AggregateNode<double>));
}

TEST_CASE("RejectIndex pops the most promising node, not the loosest bound",
          "[aggregate][selection]")
{
  // The point of the rule (§7.2), as the smallest case that distinguishes it.
  // `loose` has the lower bound but a huge objective range -- the relaxation
  // looseness §6.1 shows lb alone selects for. `tight` is genuinely promising.
  AggregateNode<double> const loose {
      .sidx = 0, .pidx = 1, .depth = 1, .lb = -1000.0, .hull_ub = 1000.0};
  AggregateNode<double> const tight {
      .sidx = 1, .pidx = 2, .depth = 1, .lb = -10.0, .hull_ub = -9.0};
  double const f_best = 0.0;

  AggregateFrontier<double> by_lb(SelectionRule::LeastLowerBound);
  by_lb.enqueue(loose, f_best);
  by_lb.enqueue(tight, f_best);
  CHECK(feq(by_lb.pop().lb, -1000.0));

  AggregateFrontier<double> by_ri(SelectionRule::RejectIndex);
  by_ri.enqueue(loose, f_best);
  by_ri.enqueue(tight, f_best);
  CHECK(feq(by_ri.pop().lb, -10.0));

  // Whichever is explored first, the reported dual bound is the same.
  CHECK(feq(by_ri.min_lb(), -1000.0));
}

TEST_CASE("RejectIndex is invariant to a positive affine rescaling of the "
          "objective",
          "[aggregate][selection]")
{
  // RI = (f - lb) / (ub - lb). Under f -> a*f + b with a > 0 the offset
  // cancels in both differences and the scale cancels between them, so the
  // rule cannot depend on the units the objective happens to be in.
  std::mt19937 rng {7};
  std::uniform_real_distribution<double> v(-100.0, 100.0);
  std::uniform_real_distribution<double> pos(0.01, 100.0);

  for (int i = 0; i < 5000; ++i) {
    double const lb = v(rng);
    double const ub = lb + pos(rng);
    double const f_best = lb + pos(rng);
    double const a = pos(rng);
    double const b = v(rng);

    double const plain =
        selection_key<double>(SelectionRule::RejectIndex, lb, ub, f_best);
    double const scaled = selection_key<double>(
        SelectionRule::RejectIndex, a * lb + b, a * ub + b, a * f_best + b);
    REQUIRE(std::abs(plain - scaled) < 1e-9 * (1.0 + std::abs(plain)));
  }
}

TEST_CASE("The selection rules' degenerate cases behave as specified",
          "[aggregate][selection]")
{
  using cuminlp::aggregate::parse_selection_rule;
  using cuminlp::aggregate::to_string;
  constexpr double inf = std::numeric_limits<double>::infinity();

  // §7.5, no incumbent: RI is +inf for everything and orders nothing, so the
  // key falls back to lb -- and must agree with LeastLowerBound exactly, or
  // the pre-incumbent phase of a reject-index run is a third, unnamed rule.
  for (double lb : {-5.0, 0.0, 12.5}) {
    CHECK(feq(selection_key<double>(
                  SelectionRule::RejectIndex, lb, lb + 3.0, kNoIncumbent),
              selection_key<double>(
                  SelectionRule::LeastLowerBound, lb, lb + 3.0, kNoIncumbent)));
  }

  // §7.5, degenerate objective interval: maximally promising while it can
  // still beat the incumbent, dominated once it cannot.
  CHECK(feq(selection_key<double>(SelectionRule::RejectIndex, 1.0, 1.0, 5.0),
            -inf));
  CHECK(feq(selection_key<double>(SelectionRule::RejectIndex, 9.0, 9.0, 5.0),
            inf));

  // A node whose lb is above the incumbent sorts after every viable one, with
  // no special case needed: the numerator goes negative.
  CHECK(selection_key<double>(SelectionRule::RejectIndex, 10.0, 20.0, 5.0)
        > selection_key<double>(SelectionRule::RejectIndex, 1.0, 20.0, 5.0));

  // Only RejectIndex re-keys; the stale variant is defined by not doing so.
  CHECK(rebuilds_on_incumbent(SelectionRule::RejectIndex));
  CHECK_FALSE(rebuilds_on_incumbent(SelectionRule::RejectIndexStale));
  CHECK_FALSE(rebuilds_on_incumbent(SelectionRule::LeastLowerBound));

  // The CLI spellings round-trip; MINLP_STATUS.md records them in a
  // reproduction key, so a silent rename would orphan recorded rows.
  for (SelectionRule rule : {SelectionRule::LeastLowerBound,
                             SelectionRule::RejectIndex,
                             SelectionRule::RejectIndexStale})
  {
    auto const parsed = parse_selection_rule(to_string(rule));
    REQUIRE(parsed);
    CHECK(*parsed == rule);
  }
  CHECK_FALSE(parse_selection_rule("reject_index"));
}

TEST_CASE("Re-keying on an improved incumbent flips the order when it should",
          "[aggregate][frontier]")
{
  // §7.4: a decrease in f_best does not shift keys uniformly -- a node with a
  // narrow objective range moves more than one with a wide range -- so after
  // an improvement the old heap is *invalid*, not merely stale.
  //
  // The pair below is chosen so the order genuinely inverts, which is the
  // only thing that distinguishes re-keying from not bothering:
  //
  //   wide:   lb = 0,  hull_ub = 100   RI = f / 100
  //   narrow: lb = 50, hull_ub = 51    RI = (f - 50) / 1
  //
  // narrow leads while f > 50.5; wide leads for f in (50, 50.5).
  AggregateNode<double> const wide {
      .sidx = 0, .pidx = 1, .depth = 1, .lb = 0.0, .hull_ub = 100.0};
  AggregateNode<double> const narrow {
      .sidx = 1, .pidx = 2, .depth = 1, .lb = 50.0, .hull_ub = 51.0};

  AggregateFrontier<double> frontier(SelectionRule::RejectIndex);
  frontier.enqueue(wide, 1000.0);
  frontier.enqueue(narrow, 1000.0);
  frontier.on_incumbent(50.2);
  CHECK(feq(frontier.pop().lb, 0.0));  // wide, on the re-keyed order

  // Same inputs, same improvement, keys frozen at insertion: the stale rule
  // must still be on the f_best = 1000 order. That the two disagree here is
  // what makes reject-index-stale a distinct thing to measure rather than an
  // implementation detail.
  AggregateFrontier<double> stale(SelectionRule::RejectIndexStale);
  stale.enqueue(wide, 1000.0);
  stale.enqueue(narrow, 1000.0);
  stale.on_incumbent(50.2);  // a no-op by construction
  CHECK(feq(stale.pop().lb, 50.0));  // narrow, on the pre-improvement order

  // Without the improvement, RejectIndex agrees with the stale rule -- so the
  // difference above is the re-keying and not some other divergence.
  AggregateFrontier<double> unimproved(SelectionRule::RejectIndex);
  unimproved.enqueue(wide, 1000.0);
  unimproved.enqueue(narrow, 1000.0);
  CHECK(feq(unimproved.pop().lb, 50.0));
}

TEST_CASE("count_viable and min_lb answer over the live set only",
          "[aggregate][frontier]")
{
  AggregateFrontier<double> frontier(SelectionRule::RejectIndex);
  double const f_best = 100.0;
  for (int i = 0; i < 10; ++i) {
    frontier.enqueue(
        AggregateNode<double> {.sidx = 0,
                               .pidx = static_cast<std::size_t>(i + 1),
                               .depth = 1,
                               .lb = static_cast<double>(i),
                               .hull_ub = static_cast<double>(i) + 1.0},
        f_best);
  }
  CHECK(frontier.count_viable(4.0) == 5);  // lb 0..4
  CHECK(feq(frontier.min_lb(), 0.0));
  CHECK(frontier.snapshot().size() == 10);

  // Popping must move min_lb even though the pop went through the *key*
  // index, leaving a stale reference in the lb index -- that discard is the
  // lazy-deletion path min_lb owns.
  while (frontier.size() > 5) {
    frontier.pop();
  }
  auto const live = frontier.snapshot();
  REQUIRE(live.size() == 5);
  CHECK(frontier.count_viable(-1.0) == 0);

  double expected = std::numeric_limits<double>::max();
  for (const auto& n : live) {
    expected = std::min(expected, n.lb);
  }
  CHECK(feq(frontier.min_lb(), expected));
}

TEST_CASE("An empty frontier reports no lower bound rather than a stale one",
          "[aggregate][frontier]")
{
  // finalise_bounds distinguishes "no pending region" from "a weak one", so
  // the sentinel matters: it feeds the clamp that decides the reported GLB.
  AggregateFrontier<double> frontier(SelectionRule::LeastLowerBound);
  CHECK(frontier.empty());
  CHECK(feq(frontier.min_lb(), std::numeric_limits<double>::max()));

  frontier.enqueue(
      AggregateNode<double> {
          .sidx = 0, .pidx = 1, .depth = 1, .lb = -3.0, .hull_ub = 2.0},
      kNoIncumbent);
  CHECK(feq(frontier.min_lb(), -3.0));
  (void)frontier.pop();
  CHECK(frontier.empty());
  CHECK(feq(frontier.min_lb(), std::numeric_limits<double>::max()));
}
