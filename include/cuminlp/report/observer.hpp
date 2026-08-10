#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/config/resolve.hpp"
#include "cuminlp/search/budget.hpp"

// SearchDriver's progress narration, pulled out from behind std::cout
// (design/MODULE_REFACTOR.md §8). A driver holds one SearchObserver and
// calls it at named points instead of writing to stdout directly; the base
// class's every override is a no-op, so a driver constructed without one is
// silent. ConsoleReporter is the one that gets today's lines back, verbatim.
namespace cuminlp::report
{

// Defined in report/telemetry.hpp; forward-declared here so on_telemetry
// below can take it by reference without this header including telemetry.hpp
// back (telemetry.hpp's TelemetryReporter derives from SearchObserver, so
// the include has to run the other way -- see that header's top comment).
struct RunTelemetry;

/// gub/candidate as reported by the ordinary (non-fathom) branch of one
/// iteration.
struct IterationEvent
{
  double gub = 0;
  double candidate = 0;
};

/// Before/after a frontier compaction, and what it left live.
struct CompactionEvent
{
  std::size_t before = 0;
  std::size_t after = 0;
  std::size_t live_bytes = 0;
  std::size_t host_budget = 0;
};

/// The search stopped because it hit the host budget -- `after_compaction`
/// distinguishes the two spellings: stopping outright (StopAtBudget, or
/// Compact's own budget still not enough after compacting) prints a
/// different sentence from the compaction event above it.
struct BudgetStopEvent
{
  bool after_compaction = false;
  std::size_t host_budget = 0;
  std::size_t pending_size = 0;
};

/// Everything the `------------ Finished ------------` block and its
/// trailing counters need. Carries `iter_idx`/`iter_limit` (not just
/// `proven_optimal`) because one of its sentences distinguishes proving
/// optimality from exhausting the iteration budget onto an already-dominated
/// frontier -- the same fact, worded two ways.
struct FinalReport
{
  double glb = 0;
  double gub = 0;
  bool proven_optimal = false;
  bool infeasible = false;
  search::StopReason stop_reason = search::StopReason::IterationLimit;
  std::size_t pending_size = 0;
  std::size_t viable = 0;
  std::size_t pruned_infeasible = 0;
  search::DropAccounting dropped;
  std::uint32_t iter_idx = 0;
  std::uint32_t iter_limit = 0;
};

/// Named points in one solve(), independent of how (or whether) they are
/// reported. `on_incumbent` deliberately takes no witness point: the
/// interface is not templated on T (SearchDriver<T> is; this observer is
/// shared by every instantiation), and nothing needs to report the point's
/// coordinates today -- ConsoleReporter's `iter N: GUB = ...` line already
/// carries the value, and a future consumer of `best_point_` can read it off
/// SolveOutcome::best_point once solve() returns instead.
struct SearchObserver
{
  virtual ~SearchObserver() = default;

  virtual void on_start([[maybe_unused]] std::size_t host_budget,
                        [[maybe_unused]] bool host_budget_requested,
                        [[maybe_unused]] bool compacting)
  {
  }

  virtual void on_dequeue([[maybe_unused]] std::uint32_t iter,
                          [[maybe_unused]] double lb)
  {
  }

  virtual void on_incumbent([[maybe_unused]] double gub) {}

  virtual void on_iteration(const IterationEvent&) {}

  virtual void on_fathom([[maybe_unused]] std::size_t n_points,
                         [[maybe_unused]] double gub)
  {
  }

  virtual void on_cache_eviction() {}

  virtual void on_compaction(const CompactionEvent&) {}

  virtual void on_budget_stop(const BudgetStopEvent&) {}

  virtual void on_backend_error(const backend::OverBudget&) {}

  virtual void on_mid_search_error([[maybe_unused]] std::string_view message) {}

  /// Fires once, immediately before on_finish -- see report/telemetry.hpp's
  /// RunTelemetry for the payload (design/TELEMETRY.md §3.2). Deliberately
  /// the only telemetry event: there is no per-prune/per-graph event, since
  /// the driver already accumulates those counts in a local and hands them
  /// over once rather than paying a virtual call in the hottest loop in the
  /// solver.
  virtual void on_telemetry(const RunTelemetry&) {}

  virtual void on_finish(const FinalReport&) {}
};

/// Reproduces today's driver output, byte-for-byte (invariant register #16):
/// tools/minlp_status.py and BOUNDED_FRONTIER.md §7 both depend on these
/// exact spellings. Deliberately narrower than the full `FinalReport`: the
/// subdomain counters (`pending_size`, `viable`, `pruned_infeasible`,
/// `dropped.dominated`) are `report::RunTelemetry`'s job now
/// (design/TELEMETRY.md) -- this prints only the bracket, why the run
/// stopped, the `Outcome:` classification tools/minlp_status.py reads when
/// no incumbent was found, and what `--bounded-frontier` cost the reported
/// bound.
class ConsoleReporter : public SearchObserver
{
public:
  /// `problem` is what `on_backend_error` costs its advice against
  /// (`config::explain_over_budget` needs a ProblemProfile, and the backend
  /// error it explains carries none) -- the same profile solve.cu already
  /// computes once per parsed model, passed through here instead of
  /// recomputed.
  explicit ConsoleReporter(config::ProblemProfile problem, bool verbose = true)
      : problem_(problem), verbose_(verbose)
  {
  }

  void on_start(std::size_t host_budget,
                bool host_budget_requested,
                bool compacting) override
  {
    if (host_budget == 0) {
      std::cout << "Host memory budget: none (nothing requested, and "
                   "MemAvailable could not be read from /proc/meminfo)\n";
    } else {
      std::cout << "Host memory budget: " << host_budget << " bytes ("
                << (host_budget_requested ? "requested" : "measured")
                << "), on reaching it: "
                << (compacting ? "compact the frontier and continue"
                               : "stop with the frontier intact")
                << '\n';
    }
  }

  void on_dequeue(std::uint32_t iter, double lb) override
  {
    last_iter_ = iter;
    print_this_iter_ = (iter % 100 == 0);
    if (verbose_ || print_this_iter_) {
      std::cout << "Least pending lb (selected for sample+partition): " << lb
                << std::endl;
    }
  }

  void on_iteration(const IterationEvent& event) override
  {
    if (verbose_ || print_this_iter_) {
      std::cout << "iter " << last_iter_ << ": GUB = " << event.gub
                << ", Candidate: " << event.candidate << std::endl;
    }
  }

  void on_fathom(std::size_t n_points, double gub) override
  {
    if (verbose_ || print_this_iter_) {
      std::cout << "iter " << last_iter_ << ": fully enumerated and fathomed ("
                << n_points << " points), GUB = " << gub << std::endl;
    }
  }

  void on_cache_eviction() override
  {
    // Printed once per run, on the first eviction only -- BackendCache's own
    // evictions_ counter (SolveOutcome::counters.cache_evictions) still
    // tallies every one of them; only the narration is once-only.
    if (printed_cache_eviction_) {
      return;
    }
    printed_cache_eviction_ = true;
    std::cout << "Device graph cache full: evicting the least recently "
                 "used composition's graphs and rebuilding as the search "
                 "revisits them.\n";
  }

  void on_compaction(const CompactionEvent& event) override
  {
    std::cout << "Host budget reached: compacted the frontier from "
              << event.before << " to " << event.after << " regions ("
              << event.live_bytes << " of " << event.host_budget
              << " bytes live)\n";
  }

  void on_budget_stop(const BudgetStopEvent& event) override
  {
    if (!event.after_compaction) {
      std::cout << "Host budget of " << event.host_budget << " bytes reached ("
                << event.pending_size
                << " regions pending); stopping the search here with the "
                   "frontier intact. The bracket below is valid. Pass "
                   "--bounded-frontier to discard the worst-bounded regions "
                   "and keep searching instead.\n";
    } else {
      std::cout << "Host budget still exceeded with the frontier "
                   "compacted; stopping the search here. The bracket "
                   "below is valid, and the regions dropped to get under "
                   "the budget are accounted for in it.\n";
    }
  }

  void on_backend_error(const backend::OverBudget& over) override
  {
    std::cout << "Out of device memory mid-search: "
              << config::explain_over_budget(over, problem_) << '\n';
  }

  void on_mid_search_error(std::string_view message) override
  {
    std::cout << message << '\n';
  }

  void on_finish(const FinalReport& report) override
  {
    if (report.infeasible) {
      std::cout
          << "Search space exhausted: every region was proven to hold no "
             "feasible point, so the problem is infeasible as far as "
             "interval analysis can tell. Sampling that merely never "
             "landed on a feasible point would have left regions pending "
             "here instead.\n";
    }
    std::cout << "------------ Finished ------------" << '\n'
              << report.glb << " <= min <= " << report.gub << '\n';
    if (report.proven_optimal) {
      std::cout << "Proven optimal: no pending region can beat the incumbent"
                << (report.iter_idx >= report.iter_limit
                        ? " (the iteration limit was reached, but every"
                          " remaining region is already dominated)"
                        : "")
                << ".\n";
    }
    std::cout << "Stop reason: " << search::to_string(report.stop_reason)
              << '\n';
    // Meaningful only when no feasible point was ever sampled -- an explicit,
    // machine-readable classification of the two ways that can happen
    // (tools/minlp_status.py's INFEASIBLE/NO_SAMPLE), printed unconditionally
    // so a plain run needs no --verbose to be recorded correctly. Replaces
    // inferring the same fact from `Pending size:`, which moved to the
    // telemetry block along with the rest of the subdomain counters.
    bool const found_incumbent = report.gub < std::numeric_limits<double>::max();
    if (!found_incumbent) {
      std::cout << "Outcome: " << (report.infeasible ? "infeasible" : "no sample")
                << '\n';
    }
    std::cout << "Dropped viable regions: " << report.dropped.viable << '\n';
    std::cout << "Dropped lb floor: ";
    if (report.dropped.lost_nothing()) {
      std::cout << "none\n";
    } else {
      std::cout << report.dropped.lb_min << '\n';
    }
  }

private:
  config::ProblemProfile problem_;
  std::uint32_t last_iter_ = 0;
  bool print_this_iter_ = false;
  bool printed_cache_eviction_ = false;
  bool verbose_ = true;
};

}  // namespace cuminlp::report
