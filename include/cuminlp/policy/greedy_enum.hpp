#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/bisection_budget.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/policy/width_first.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"

namespace cuminlp::policy
{

// Greedy, stateless composition policy: fills slots from unresolved (i.e.
// non-degenerate, lb < ub) variables, binaries first, then integers, then
// continuous.
//
// Owns its own FanOutSpec (design/BUDGETED_PARTITION.md §14.3): the type
// stays exactly what it always was, uniform width per SlotKind, but it is no
// longer part of the CompositionPolicy base interface or the decode/backend
// pipeline -- BisectionBudgetCompositionPolicy has no single such spec to
// share, so each policy that needs one now keeps it privately and fills
// SlotAssignment::widths explicitly in choose().
template<typename T>
class GreedyEnumCompositionPolicy : public CompositionPolicy<T>
{
public:
  explicit GreedyEnumCompositionPolicy(
      region::FanOutSpec fan_out, config::SearchCalibration calibration = {})
      : CompositionPolicy<T>(calibration)
      , fan_out_(fan_out)
  {
  }

  using CompositionPolicy<T>::max_cycle_size;

  const region::FanOutSpec& fan_out() const { return fan_out_; }

  std::size_t n_regions(const region::Composition& composition) const override
  {
    return region::composition_fan_out(composition, fan_out_);
  }

  region::SlotAssignment choose(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const override
  {
    assert(box.size() == var_kinds.size());
    if (var_kinds.empty()) {
      // Nothing to assign var_ids[s] = 0 to -- 0 isn't a valid variable
      // index on a zero-variable problem (TEST_EXTENSION.md, the
      // "every var_id < var_kinds.size()" invariant).
      throw ShapeMismatch(
          "CompositionPolicy::choose called with an empty var_kinds span; "
          "there are no " "variables to assign to any slot");
    }

    region::SlotAssignment out {};

    fill_binary(box, var_kinds, out);
    fill_integer(box, var_kinds, out);
    fill_continuous(box, var_kinds, out);

    return out;
  }

  // Number of integers in [ceil(b.lb), floor(b.ub)]. b need not itself be
  // lattice-aligned -- reachable directly from an IntegerPartition child,
  // whose boundaries reuse the continuous linear-width formula
  // (TEST_EXTENSION.md) -- so ceil/floor do the snapping here rather than
  // assuming the caller already aligned them. Returns 0, rather than
  // underflowing (UB, in fact: casting a negative double to size_t) a huge
  // size_t, when the sub-box contains no integer at all (ceil(b.lb) >
  // floor(b.ub)): this can't be enumerated or partitioned further and is
  // deterministically empty. Public (rather than an implementation-detail
  // private helper) because it's the exact locus TEST_EXTENSION.md calls out
  // by name.
  static std::size_t integer_domain_size(const cu::interval<T>& b)
  {
    T const lo = std::ceil(b.lb);
    T const hi = std::floor(b.ub);
    if (hi < lo) {
      return 0;
    }
    return static_cast<std::size_t>(hi - lo) + 1;
  }

private:
  static bool unresolved(const cu::interval<T>& b) { return b.ub > b.lb; }

  void append(region::SlotAssignment& out,
              region::SlotKind kind,
              std::size_t vid) const
  {
    out.composition.kinds.push_back(kind);
    out.var_ids.push_back(vid);
    out.widths.push_back(region::slot_fan_out(kind, fan_out_));
  }

  bool at_cap(const region::SlotAssignment& out) const
  {
    return out.var_ids.size() >= max_cycle_size();
  }

  // All three fill helpers stop at the policy's cap: a composition is
  // exactly its live slots, so a box with more live variables than the cap
  // simply leaves the rest unassigned this iteration.
  void fill_binary(std::span<const cu::interval<T>> box,
                   std::span<const model::VarKind> var_kinds,
                   region::SlotAssignment& out) const
  {
    for (std::size_t vid = 0; vid < var_kinds.size() && !at_cap(out); ++vid) {
      if (var_kinds[vid] == model::VarKind::Binary && unresolved(box[vid])) {
        append(out, region::SlotKind::BinaryEnumerate, vid);
      }
    }
  }

  // Non-static, unlike its binary/continuous siblings: it's the one that
  // consults enumerate_cap, which now lives on the policy instance.
  void fill_integer(std::span<const cu::interval<T>> box,
                    std::span<const model::VarKind> var_kinds,
                    region::SlotAssignment& out) const
  {
    std::vector<std::size_t> candidates;
    for (std::size_t vid = 0; vid < var_kinds.size(); ++vid) {
      if (var_kinds[vid] == model::VarKind::Integer && unresolved(box[vid])) {
        candidates.push_back(vid);
      }
    }
    // Smallest remaining domain first: most likely to become enumerable (or
    // already is), so this makes the fastest progress toward fully
    // resolving a dimension.
    std::sort(
        candidates.begin(),
        candidates.end(),
        [&](std::size_t a, std::size_t b)
        { return integer_domain_size(box[a]) < integer_domain_size(box[b]); });

    std::size_t const enumerate_cap = fan_out().enumerate_cap();
    for (std::size_t vid : candidates) {
      if (at_cap(out)) {
        break;
      }
      region::SlotKind const kind =
          integer_domain_size(box[vid]) <= enumerate_cap
          ? region::SlotKind::IntegerEnumerate
          : region::SlotKind::IntegerPartition;
      append(out, kind, vid);
    }
  }

  void fill_continuous(std::span<const cu::interval<T>> box,
                       std::span<const model::VarKind> var_kinds,
                       region::SlotAssignment& out) const
  {
    std::vector<std::size_t> candidates;
    for (std::size_t vid = 0; vid < var_kinds.size(); ++vid) {
      if (var_kinds[vid] == model::VarKind::Continuous && unresolved(box[vid]))
      {
        candidates.push_back(vid);
      }
    }
    // Widest first: classic largest-uncertainty-first bisection heuristic.
    std::sort(candidates.begin(),
              candidates.end(),
              [&](std::size_t a, std::size_t b)
              { return (box[a].ub - box[a].lb) > (box[b].ub - box[b].lb); });

    for (std::size_t vid : candidates) {
      if (at_cap(out)) {
        break;
      }
      append(out, region::SlotKind::Continuous, vid);
    }
  }

  region::FanOutSpec fan_out_;
};

/**
 * @brief Construct the CompositionPolicy subclass a PolicyKind names.
 *
 * A switch, not an if-chain, is deliberate: adding a PolicyKind without a
 * matching case here is a compiler warning (-Wswitch) away from being caught.
 *
 * `fan_out` is what GreedyEnumerate/WidthFirst construct against;
 * `bisection_budget` is BisectionBudgetCompositionPolicy's `B`
 * (design/BUDGETED_PARTITION.md) -- each PolicyKind reads only the one it
 * needs, so callers building a RunSpec fill in both without having to know
 * in advance which the chosen policy will use.
 */
template<typename T>
std::unique_ptr<CompositionPolicy<T>> make_policy(
    PolicyKind kind,
    region::FanOutSpec fan_out,
    std::size_t bisection_budget,
    config::SearchCalibration calibration)
{
  switch (kind) {
    case PolicyKind::GreedyEnumerate:
      return std::make_unique<GreedyEnumCompositionPolicy<T>>(fan_out, calibration);
    case PolicyKind::WidthFirst:
      return std::make_unique<WidthFirstCompositionPolicy<T>>(fan_out, calibration);
    case PolicyKind::BisectionBudget:
      return std::make_unique<BisectionBudgetCompositionPolicy<T>>(
          bisection_budget, calibration);
  }
  throw InvalidConfiguration("unknown PolicyKind");
}

}  // namespace cuminlp::policy
