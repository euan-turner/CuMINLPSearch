#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

// Which pending node the aggregate search explores next
// (design/AGGREGATE_BOUNDING.md §7). A pure function of a node's bounds and
// the current incumbent, deliberately separated from the frontier that
// applies it so the rules are testable without a queue and swappable without
// touching one.
//
// Host-only; no CUDA type is in reach.
namespace cuminlp::aggregate
{

enum class SelectionRule
{
  /// §7.1. Best-first on the dual bound: pop the least `lb`. The default, and
  /// what the existing `search::IntervalPQueue` already does.
  LeastLowerBound,

  /// §7.2. Pop the largest `(f_best - lb) / (hull_ub - lb)`. Normalises out
  /// the relaxation looseness that `lb` alone selects for. Keys are recomputed
  /// whenever the incumbent improves (§7.4).
  RejectIndex,

  /// RejectIndex with each node's key fixed at insertion. **For measurement
  /// only** (§7.4): it is what most implementations do, but it makes which
  /// tree the search explored depend on incumbent timing, which would muddy
  /// the very A/B this backend exists to run.
  RejectIndexStale,
};

inline const char* to_string(SelectionRule rule)
{
  switch (rule) {
    case SelectionRule::LeastLowerBound:
      return "least-lower-bound";
    case SelectionRule::RejectIndex:
      return "reject-index";
    case SelectionRule::RejectIndexStale:
      return "reject-index-stale";
  }
  return "?";
}

inline std::optional<SelectionRule> parse_selection_rule(std::string_view name)
{
  if (name == "least-lower-bound") {
    return SelectionRule::LeastLowerBound;
  }
  if (name == "reject-index") {
    return SelectionRule::RejectIndex;
  }
  if (name == "reject-index-stale") {
    return SelectionRule::RejectIndexStale;
  }
  return std::nullopt;
}

/// True iff a node's key can change when the incumbent improves, and the
/// frontier must therefore re-key and re-heapify (§7.4). False for
/// `LeastLowerBound` (whose key is just `lb`) and, by construction, for
/// `RejectIndexStale`.
inline bool rebuilds_on_incumbent(SelectionRule rule)
{
  return rule == SelectionRule::RejectIndex;
}

/**
 * @brief A node's priority. **Smaller is explored sooner**, for every rule.
 *
 * One sign convention across all rules, so the frontier's comparator never
 * branches on which rule it holds: `LeastLowerBound` returns `lb` directly,
 * and the RejectIndex rules return `-RI`, since a *large* RI is promising.
 *
 * The degenerate cases are §7.5, and each is a real reachable state rather
 * than defensive padding:
 *
 * - **No incumbent yet** (`f_best` infinite). `RI` is `+inf` for every node
 *   and orders nothing, so the key falls back to `lb`. On an equality-heavy
 *   model the sampler may never produce an incumbent at all (§6.4), so this
 *   fallback can be the entire run -- which is why the driver reports whether
 *   it was ever left, rather than reporting the rule that was merely asked
 *   for.
 * - **Degenerate objective interval** (`hull_ub == lb`). The child is fully
 *   resolved with respect to the objective. If it can still beat the
 *   incumbent it is maximally promising and sorts first; otherwise it is
 *   dominated and sorts last. Note the second branch is a safety net, not a
 *   path: the driver's cutoff discards such a node before it is ever
 *   enqueued.
 * - **`lb` above the incumbent.** The numerator goes negative, so the key
 *   goes positive and the node sorts after every viable one. Nothing special
 *   is needed for it.
 *
 * `hull_ub < lb` is not handled: it cannot arise from two reductions over the
 * same mask, and the frontier asserts against it rather than defining a
 * meaning for it.
 */
template<typename T>
double selection_key(SelectionRule rule, T lb, T hull_ub, double f_best)
{
  double const lb_d = static_cast<double>(lb);
  if (rule == SelectionRule::LeastLowerBound) {
    return lb_d;
  }

  // No incumbent: RI is uninformative, so order by the dual bound instead.
  if (!(f_best < std::numeric_limits<double>::max())) {
    return lb_d;
  }

  double const width = static_cast<double>(hull_ub) - lb_d;
  if (!(width > 0.0)) {
    return lb_d < f_best ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity();
  }

  return -((f_best - lb_d) / width);
}

}  // namespace cuminlp::aggregate
