#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/composition.hpp"

// design/BUDGETED_PARTITION.md's BisectionBudget policy (§7): a fixed
// per-solve region budget N = 2^B, spent per node minimising the widest
// live domain after the split. Unlike GreedyEnumerate/WidthFirst, it has no
// shared FanOutSpec -- every slot's width is a per-node decision, chosen
// here and carried out on SlotAssignment::widths -- so `N` is a pure
// function of B alone (CompositionPolicy::n_regions), not of Composition.
namespace cuminlp::policy
{

template<typename T>
class BisectionBudgetCompositionPolicy : public CompositionPolicy<T>
{
public:
  // `bisection_budget` (`B`, §1.1) must be at least 1 (a slot needs b_j >= 1
  // to exist at all, §8.2) and small enough that 2^B fits a 64-bit region
  // count with headroom -- 62 rather than 63/64 is a guard against
  // absurdity, the same spirit as config::kMaxSlots, not a measured limit.
  explicit BisectionBudgetCompositionPolicy(
      std::size_t bisection_budget,
      config::SearchCalibration calibration = {})
      : CompositionPolicy<T>(calibration)
      , B_(bisection_budget)
  {
    if (B_ < 1 || B_ > 62) {
      throw InvalidConfiguration(
          "bisection_budget must be between 1 and 62; B >= 1 guarantees at "
          "least one bisection is always available, and 2^B must fit "
          "comfortably in a 64-bit region count");
    }
    N_ = std::size_t {1} << B_;
  }

  using CompositionPolicy<T>::max_cycle_size;

  std::size_t bisection_budget() const { return B_; }

  // A pure function of B alone: every composition this policy ever returns
  // costs exactly N regions (design/BUDGETED_PARTITION.md §3), which is the
  // entire point -- except the empty composition a fully-resolved box
  // produces, which is the identity's empty product (1), not N. The driver
  // never actually asks the backend to build a subdividing graph for that
  // case (can_fathom_without_children takes the enumerator path first), but
  // n_regions() must still answer it correctly.
  std::size_t n_regions(const region::Composition& composition) const override
  {
    return composition.size() == 0 ? std::size_t {1} : N_;
  }

  region::SlotAssignment choose(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const override
  {
    assert(box.size() == var_kinds.size());
    if (var_kinds.empty()) {
      throw ShapeMismatch(
          "CompositionPolicy::choose called with an empty var_kinds span; "
          "there are no variables to assign to any slot");
    }

    // Live variables that will *enumerate* if the budget admits them
    // (Binary always, Integer when its domain is under the enumerate_cap
    // ceiling, if any) -- `cost` is `ceil(log2(max(domain, 1)))`, floored at
    // 1 so an emitted slot always has b_j >= 1 (§5's main-line rule).
    struct EnumCandidate
    {
      std::size_t vid;
      region::SlotKind kind;
      std::size_t domain;  // true domain size d_j
      std::size_t bisections;  // ceil(log2(max(d_j, 1))), floored at 1
    };
    std::vector<EnumCandidate> enum_candidates;

    // Live variables that will *bisect*: every Continuous variable, plus any
    // Integer that is either over the enumerate_cap ceiling or misses out on
    // the budget below (§8.2's fallback).
    struct BisectCandidate
    {
      std::size_t vid;
      region::SlotKind kind;
      T live_width;  // current live domain width; halved on each pick
      std::size_t bisections = 0;
    };
    std::vector<BisectCandidate> bisect_candidates;

    std::size_t const cap = max_cycle_size();
    std::size_t const enumerate_ceiling = this->calibration().enumerate_cap;

    for (std::size_t vid = 0;
        vid < var_kinds.size()
        && enum_candidates.size() + bisect_candidates.size() < cap;
        ++vid)
    {
      const cu::interval<T>& b = box[vid];
      if (!(b.ub > b.lb)) {
        continue;  // resolved; nothing to do this cycle
      }
      switch (var_kinds[vid]) {
        case model::VarKind::Binary:
          enum_candidates.push_back(
              {vid, region::SlotKind::BinaryEnumerate, 2, bits_for(2)});
          break;
        case model::VarKind::Integer: {
          std::size_t const d = integer_domain_size(b);
          bool const under_ceiling =
              enumerate_ceiling == 0 || std::max<std::size_t>(d, 1) <= enumerate_ceiling;
          if (under_ceiling) {
            enum_candidates.push_back(
                {vid, region::SlotKind::IntegerEnumerate, d, bits_for(d)});
          } else {
            bisect_candidates.push_back(
                {vid, region::SlotKind::IntegerPartition, width(b)});
          }
          break;
        }
        case model::VarKind::Continuous:
          bisect_candidates.push_back(
              {vid, region::SlotKind::Continuous, width(b)});
          break;
      }
    }

    if (enum_candidates.empty() && bisect_candidates.empty()) {
      return region::SlotAssignment {};  // fully resolved box
    }

    // §8.2: enumerate slots taken cheapest-first until the next would
    // exceed B. Sorted ascending by cost, so once one candidate doesn't fit,
    // no later one (equal or higher cost) does either -- a stop, not a skip.
    std::stable_sort(enum_candidates.begin(),
                     enum_candidates.end(),
                     [](const EnumCandidate& a, const EnumCandidate& b)
                     { return a.bisections < b.bisections; });

    std::size_t remaining = B_;
    std::size_t taken_count = 0;
    for (; taken_count < enum_candidates.size(); ++taken_count) {
      if (enum_candidates[taken_count].bisections > remaining) {
        break;
      }
      remaining -= enum_candidates[taken_count].bisections;
    }

    // Not-taken Integers fall back to IntegerPartition -- what they would
    // have got anyway had their domain been over the ceiling (§8.2).
    // Not-taken Binaries simply get no slot this cycle: there is no
    // BinaryPartition kind, the same as GreedyEnumerate/WidthFirst.
    for (std::size_t i = taken_count; i < enum_candidates.size(); ++i) {
      const EnumCandidate& c = enum_candidates[i];
      if (c.kind == region::SlotKind::IntegerEnumerate) {
        bisect_candidates.push_back(
            {c.vid, region::SlotKind::IntegerPartition, width(box[c.vid])});
      }
    }

    // §7: greedy max-heap by absolute live width, B times (here `remaining`
    // times, since taking an enumerate slot already spent some of B).
    // Exactly optimal for the min-max objective (the exchange argument in
    // the design doc): moving one bisection from a non-maximal variable to
    // the maximal one weakly improves it, so iterating settles on greedy.
    std::vector<std::size_t> heap(bisect_candidates.size());
    std::iota(heap.begin(), heap.end(), 0);
    auto by_live_width = [&bisect_candidates](std::size_t a, std::size_t b)
    { return bisect_candidates[a].live_width < bisect_candidates[b].live_width; };
    std::make_heap(heap.begin(), heap.end(), by_live_width);

    std::size_t spent_bisecting = 0;
    for (; spent_bisecting < remaining && !heap.empty(); ++spent_bisecting) {
      std::pop_heap(heap.begin(), heap.end(), by_live_width);
      BisectCandidate& c = bisect_candidates[heap.back()];
      c.live_width = c.live_width / T {2};
      ++c.bisections;
      std::push_heap(heap.begin(), heap.end(), by_live_width);
    }

    // The only way `remaining` outlives an empty bisecting pool: every live
    // variable enumerated (e.g. an all-binary box narrower than B) and the
    // budget still has bisections left over. Widening an already-covering
    // enumerate slot buys no new information, but Π widths == N must still
    // hold, so the surplus is spent as padding on the widest taken slot --
    // duplicate parts and all, the same waste §8 already accepts, kept from
    // propagating by the same §8.1 suppression.
    if (spent_bisecting < remaining) {
      assert(taken_count > 0
            && "B >= 1 and every live variable is enumerate-type with no "
               "bisecting pool: at least one candidate must have been taken");
      for (std::size_t k = spent_bisecting; k < remaining; ++k) {
        std::size_t widest = 0;
        for (std::size_t i = 1; i < taken_count; ++i) {
          if (enum_candidates[i].bisections > enum_candidates[widest].bisections)
          {
            widest = i;
          }
        }
        ++enum_candidates[widest].bisections;
      }
    }

    region::SlotAssignment out {};
    for (std::size_t i = 0; i < taken_count; ++i) {
      const EnumCandidate& c = enum_candidates[i];
      out.composition.kinds.push_back(c.kind);
      out.var_ids.push_back(c.vid);
      out.widths.push_back(std::size_t {1} << c.bisections);
      out.domain_sizes.push_back(c.domain);
    }
    for (const BisectCandidate& c : bisect_candidates) {
      if (c.bisections == 0) {
        continue;  // the budget ran out before reaching it; no slot this cycle
      }
      out.composition.kinds.push_back(c.kind);
      out.var_ids.push_back(c.vid);
      std::size_t const w = std::size_t {1} << c.bisections;
      out.widths.push_back(w);
      out.domain_sizes.push_back(w);  // exact split: never a duplicate
    }

    return out;
  }

private:
  static T width(const cu::interval<T>& b) { return b.ub - b.lb; }

  // Duplicated from GreedyEnumCompositionPolicy/WidthFirstCompositionPolicy
  // rather than shared, to avoid a circular include -- see their own
  // comments for the ceil/floor and empty-domain rationale.
  static std::size_t integer_domain_size(const cu::interval<T>& b)
  {
    T const lo = std::ceil(b.lb);
    T const hi = std::floor(b.ub);
    if (hi < lo) {
      return 0;
    }
    return static_cast<std::size_t>(hi - lo) + 1;
  }

  // ceil(log2(max(d, 1))), floored at 1: the bisections an enumerate slot
  // for a domain of size d costs (design/BUDGETED_PARTITION.md §8), never
  // 0 so an emitted slot always satisfies the main-line rule (§5) of
  // b_j >= 1.
  static std::size_t bits_for(std::size_t d)
  {
    std::size_t const eff = d == 0 ? 1 : d;
    std::size_t bits = 0;
    std::size_t cap = 1;
    while (cap < eff) {
      cap <<= 1;
      ++bits;
    }
    return std::max<std::size_t>(bits, 1);
  }

  std::size_t B_;
  std::size_t N_;
};

}  // namespace cuminlp::policy
