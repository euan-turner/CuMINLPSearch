#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/budget.hpp"

// One struct the run's counters collect into, and the reporter that prints
// it (design/TELEMETRY.md). Telemetry is report's second job, alongside
// ConsoleReporter's narration -- see report/observer.hpp for the event this
// is raised through (RunTelemetry is forward-declared there, defined here)
// and report/fan.hpp for how the two reporters coexist.
namespace cuminlp::report
{

/// What one solve() counted, from dequeue to the four ways a subdomain can
/// leave the search, to the device graphs BackendCache built along the way.
/// Populated once by the driver after the loop and consumed by both
/// TelemetryReporter and library callers via SolveOutcome::counters
/// (design/TELEMETRY.md §3.1).
struct RunTelemetry
{
  // subdomain flow
  std::uint64_t iterations = 0;  ///< dequeues
  std::uint64_t fathomed = 0;  ///< dequeues that took the enumerate path
  std::uint64_t fathomed_points = 0;  ///< exact points those evaluated
  std::uint64_t bounded = 0;  ///< sum of BoundResult::n_regions
  std::uint64_t sampled = 0;  ///< sum of RegionSampler::n_samples()
  std::uint64_t enqueued = 0;
  // design/BUDGETED_PARTITION.md §6.2: a witness rejected by
  // witness_is_admissible rather than folded into GUB_.
  std::uint64_t witness_rejected = 0;
  // design/BUDGETED_PARTITION.md §8.1: a duplicate child (a rounded-up
  // enumerate slot's digit landing past its true domain) excluded before
  // enqueueing rather than explored again.
  std::uint64_t pruned_duplicate = 0;

  // why a subdomain left, all four ways
  std::uint64_t pruned_infeasible = 0;
  std::uint64_t pruned_dominated = 0;  ///< dominated at insertion
  search::DropAccounting dropped;  ///< the sweep: .dominated / .viable / .lb_min

  // frontier and history
  std::size_t pending = 0;
  std::size_t viable = 0;
  std::uint64_t history_freed = 0;  ///< refcount hit zero
  std::size_t history_peak_live = 0;

  // device -- counts only; the per-graph detail is backend::graph's own line
  std::uint64_t graphs_built = 0;
  std::uint64_t graphs_rebuilt = 0;  ///< a composition+role built again after eviction
  std::size_t cache_evictions = 0;

  /// §2.1's free self-check: two identities the independently-tracked
  /// counters above must satisfy if nothing was miscounted.
  ///
  ///   bounded      = pruned_infeasible + pruned_dominated + pruned_duplicate
  ///                  + enqueued
  ///   enqueued + 1 = dequeued + swept_dominated + swept_viable + pending
  ///
  /// A duplicate child (design/BUDGETED_PARTITION.md §8.1) is excluded
  /// before the feasible[]/obj_lb[] checks, so it is still counted in
  /// `bounded` (the bounder evaluated it) but never reaches `enqueued`,
  /// `pruned_infeasible` or `pruned_dominated` -- its own term in the first
  /// identity.
  ///
  /// The `+ 1` is the root, enqueued directly rather than through the child
  /// loop. A counter that disagrees with the others is a bug report, not a
  /// rounding difference.
  bool balances() const
  {
    bool const bound_flow = bounded
        == pruned_infeasible + pruned_dominated + pruned_duplicate + enqueued;
    bool const frontier_flow = enqueued + 1
        == iterations + dropped.dominated + dropped.viable + pending;
    return bound_flow && frontier_flow;
  }
};

/// Prints the telemetry block (design/TELEMETRY.md §5.2, minus its machine-
/// readable TELEMETRY line -- every field there restated a field already in
/// the human block above it, and nothing parses it, so it was dropped rather
/// than kept as an unconsumed second copy). Holds the RunTelemetry handed to
/// it by on_telemetry and prints it from on_finish -- after ConsoleReporter's
/// block, since ObserverFan forwards in registration order and
/// TelemetryReporter is registered second.
class TelemetryReporter : public SearchObserver
{
public:
  void on_telemetry(const RunTelemetry& telemetry) override
  {
    telemetry_ = telemetry;
  }

  void on_finish(const FinalReport&) override
  {
    std::size_t const swept = telemetry_.dropped.dominated + telemetry_.dropped.viable;
    bool const ok = telemetry_.balances();

    std::cout << "----------- Telemetry -----------\n"
              << "subdomains  seen " << telemetry_.iterations << "  bounded "
              << telemetry_.bounded << "  sampled " << telemetry_.sampled
              << "  enqueued " << telemetry_.enqueued << '\n'
              << "            fathomed " << telemetry_.fathomed
              << " dequeues (" << telemetry_.fathomed_points
              << " exact points)\n"
              << "pruned      infeasible " << telemetry_.pruned_infeasible
              << "  dominated " << telemetry_.pruned_dominated
              << "  duplicate " << telemetry_.pruned_duplicate
              << "  swept-dominated " << telemetry_.dropped.dominated
              << "  swept-viable " << telemetry_.dropped.viable << '\n'
              << "witness     rejected " << telemetry_.witness_rejected << '\n'
              << "frontier    pending " << telemetry_.pending << "  viable "
              << telemetry_.viable << '\n'
              << "history     slots freed " << telemetry_.history_freed
              << "  peak live " << telemetry_.history_peak_live << '\n'
              << "graphs      built " << telemetry_.graphs_built
              << "  rebuilt " << telemetry_.graphs_rebuilt << "  evictions "
              << telemetry_.cache_evictions
              << "  (see the GRAPH lines above)\n"
              << "balance     enqueued " << telemetry_.enqueued << " + root 1 = "
              << "dequeued " << telemetry_.iterations << " + swept " << swept
              << " + pending " << telemetry_.pending << "  "
              << (ok ? "OK" : "MISMATCH") << '\n';
  }

private:
  RunTelemetry telemetry_;
};

}  // namespace cuminlp::report
