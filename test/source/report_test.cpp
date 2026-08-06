#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/report/observer.hpp"

using cuminlp::backend::OverBudget;
using cuminlp::config::ProblemProfile;
using cuminlp::report::BudgetStopEvent;
using cuminlp::report::CompactionEvent;
using cuminlp::report::ConsoleReporter;
using cuminlp::report::FinalReport;
using cuminlp::report::IterationEvent;
using cuminlp::search::DropAccounting;
using cuminlp::search::FrontierPolicy;
using cuminlp::search::StopReason;

namespace
{

// Redirects std::cout to a string for the lifetime of the object, so a
// ConsoleReporter call's exact text can be asserted against -- the same
// technique test/source/dag_test.cpp uses for print_problem.
class CoutCapture
{
public:
  CoutCapture()
      : prev_(std::cout.rdbuf(buf_.rdbuf()))
  {
  }

  ~CoutCapture() { std::cout.rdbuf(prev_); }

  std::string str() const { return buf_.str(); }

private:
  std::ostringstream buf_;
  std::streambuf* prev_;
};

}  // namespace

// ---------------------------------------------------------------------------
// on_start
// ---------------------------------------------------------------------------

TEST_CASE("on_start with a zero budget reports it could not be measured",
          "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_start(0, false, false);
  CHECK(out.str()
        == "Host memory budget: none (nothing requested, and MemAvailable "
           "could not be read from /proc/meminfo)\n");
}

TEST_CASE("on_start reports a measured budget that stops the search",
          "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_start(1000, false, false);
  CHECK(out.str()
        == "Host memory budget: 1000 bytes (measured), on reaching it: stop "
           "with the frontier intact\n");
}

TEST_CASE("on_start reports a requested budget that compacts", "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_start(2000, true, true);
  CHECK(out.str()
        == "Host memory budget: 2000 bytes (requested), on reaching it: "
           "compact the frontier and continue\n");
}

// ---------------------------------------------------------------------------
// on_dequeue / on_iteration / on_fathom -- the iter number on_dequeue
// receives but does not itself print, that on_iteration/on_fathom rely on
// ---------------------------------------------------------------------------

TEST_CASE("on_dequeue prints no iter number, but on_iteration reuses the "
          "last one it was given",
          "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_dequeue(7, -3.5);
  reporter.on_iteration(IterationEvent {.gub = 2.5, .candidate = 3.0});
  CHECK(out.str()
        == "Least pending lb (selected for sample+partition): -3.5\n"
           "iter 7: GUB = 2.5, Candidate: 3\n");
}

TEST_CASE("on_fathom reuses the last iter number from on_dequeue", "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_dequeue(12, 0.0);
  reporter.on_fathom(4, -1.25);
  CHECK(out.str()
        == "Least pending lb (selected for sample+partition): 0\n"
           "iter 12: fully enumerated and fathomed (4 points), GUB = "
           "-1.25\n");
}

// ---------------------------------------------------------------------------
// on_cache_eviction: printed once only
// ---------------------------------------------------------------------------

TEST_CASE("on_cache_eviction prints only on the first call", "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_cache_eviction();
  reporter.on_cache_eviction();
  reporter.on_cache_eviction();
  CHECK(out.str()
        == "Device graph cache full: evicting the least recently used "
           "composition's graphs and rebuilding as the search revisits "
           "them.\n");
}

// ---------------------------------------------------------------------------
// on_compaction / on_budget_stop
// ---------------------------------------------------------------------------

TEST_CASE("on_compaction reports before/after/live against the budget",
          "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_compaction(CompactionEvent {
      .before = 100, .after = 25, .live_bytes = 900, .host_budget = 1000});
  CHECK(out.str()
        == "Host budget reached: compacted the frontier from 100 to 25 "
           "regions (900 of 1000 bytes live)\n");
}

TEST_CASE("on_budget_stop before compaction points at --bounded-frontier",
          "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_budget_stop(BudgetStopEvent {
      .after_compaction = false, .host_budget = 1000, .pending_size = 42});
  CHECK(out.str()
        == "Host budget of 1000 bytes reached (42 regions pending); "
           "stopping the search here with the frontier intact. The bracket "
           "below is valid. Pass --bounded-frontier to discard the "
           "worst-bounded regions and keep searching instead.\n");
}

TEST_CASE(
    "on_budget_stop after compaction says the budget is still " "exceeded",
    "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_budget_stop(BudgetStopEvent {
      .after_compaction = true, .host_budget = 1000, .pending_size = 5});
  CHECK(out.str()
        == "Host budget still exceeded with the frontier compacted; "
           "stopping the search here. The bracket below is valid, and the "
           "regions dropped to get under the budget are accounted for in "
           "it.\n");
}

// ---------------------------------------------------------------------------
// on_backend_error / on_mid_search_error
// ---------------------------------------------------------------------------

TEST_CASE("on_backend_error prefixes explain_over_budget's report", "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  OverBudget over {
      .role = "interval graph",
      .needed_bytes = 1000,
      .budget_bytes = 10,
      .n_regions = 4,
      .elements_per_region = 1,
      .bytes_per_element = 250,
      .element_breakdown = "4 buffers",
      .element_inventory = "4 DAG nodes",
      .cost = {},
      .composition = {},
      .fan_out = cuminlp::region::FanOutSpec {2, 2},
      .solve_samples_per_region = 1,
      .sampler_bytes = 0,
  };

  CoutCapture out;
  reporter.on_backend_error(over);
  std::string const text = out.str();
  CHECK(text.starts_with(
      "Out of device memory mid-search: interval graph " "needs"));
  CHECK(text.ends_with('\n'));
}

TEST_CASE("on_mid_search_error prints the message verbatim", "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_mid_search_error("Host allocation failed mid-search: bad_alloc");
  CHECK(out.str() == "Host allocation failed mid-search: bad_alloc\n");
}

// ---------------------------------------------------------------------------
// on_finish
// ---------------------------------------------------------------------------

TEST_CASE("on_finish reports an ordinary iteration-limit stop", "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_finish(FinalReport {
      .glb = -5.0,
      .gub = -3.0,
      .proven_optimal = false,
      .infeasible = false,
      .stop_reason = StopReason::IterationLimit,
      .pending_size = 12,
      .viable = 3,
      .pruned_infeasible = 7,
      .dropped = DropAccounting {},
      .iter_idx = 1000,
      .iter_limit = 1000,
  });
  CHECK(out.str()
        == "------------ Finished ------------\n"
           "-5 <= min <= -3\n"
           "Stop reason: iteration-limit\n"
           "Pending size: 12\n"
           "Viable regions: 3\n"
           "Pruned as interval-infeasible: 7\n"
           "Dropped viable regions: 0\n"
           "Dropped dominated regions: 0\n"
           "Dropped lb floor: none\n");
}

TEST_CASE("on_finish notes when proven optimal at the iteration limit",
          "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_finish(FinalReport {
      .glb = -3.0,
      .gub = -3.0,
      .proven_optimal = true,
      .infeasible = false,
      .stop_reason = StopReason::IterationLimit,
      .pending_size = 5825,
      .viable = 0,
      .pruned_infeasible = 0,
      .dropped = DropAccounting {},
      .iter_idx = 1248,
      .iter_limit = 1248,
  });
  CHECK(
      out.str().find("Proven optimal: no pending region can beat the incumbent "
                     "(the " "iteration limit was reached, but every remaining "
                             "region is " "already dominated).\n")
      != std::string::npos);
}

TEST_CASE(
    "on_finish notes plain proven-optimal short of the iteration " "limit",
    "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_finish(FinalReport {
      .glb = -3.0,
      .gub = -3.0,
      .proven_optimal = true,
      .infeasible = false,
      .stop_reason = StopReason::Converged,
      .pending_size = 0,
      .viable = 0,
      .pruned_infeasible = 0,
      .dropped = DropAccounting {},
      .iter_idx = 10,
      .iter_limit = 1000,
  });
  CHECK(out.str().find(
            "Proven optimal: no pending region can beat the incumbent.\n")
        != std::string::npos);
}

TEST_CASE("on_finish reports infeasibility ahead of the Finished block",
          "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  CoutCapture out;
  reporter.on_finish(FinalReport {
      .glb = -1e300,
      .gub = 1e300,
      .proven_optimal = false,
      .infeasible = true,
      .stop_reason = StopReason::Exhausted,
      .pending_size = 0,
      .viable = 0,
      .pruned_infeasible = 9,
      .dropped = DropAccounting {},
      .iter_idx = 3,
      .iter_limit = 1000,
  });
  std::string const text = out.str();
  CHECK(text
            .starts_with("Search space exhausted: every region was proven to "
                         "hold no feasible " "point, so the problem is "
                                             "infeasible as far as interval "
                                             "analysis can " "tell. Sampling "
                                                             "that merely "
                                                             "never landed on "
                                                             "a feasible point "
                                                             "would " "have "
                                                                      "left "
                                                                      "regions "
                                                                      "pending "
                                                                      "here "
                                                                      "instead."
                                                                      "\n" "---"
                                                                           "---"
                                                                           "---"
                                                                           "---"
                                                                           " Fi"
                                                                           "nis"
                                                                           "hed"
                                                                           " --"
                                                                           "---"
                                                                           "---"
                                                                           "---"
                                                                           "-"
                                                                           "\n"));
}

TEST_CASE(
    "on_finish prints the dropped lb floor when something was " "dropped",
    "[report]")
{
  ConsoleReporter reporter {ProblemProfile {}};
  DropAccounting dropped;
  dropped.fold(-2.0, 5.0);  // viable: -2.0 <= 5.0
  dropped.fold(6.0, 5.0);  // dominated: 6.0 > 5.0

  CoutCapture out;
  reporter.on_finish(FinalReport {
      .glb = -2.0,
      .gub = 5.0,
      .proven_optimal = false,
      .infeasible = false,
      .stop_reason = StopReason::HostMemory,
      .pending_size = 1,
      .viable = 1,
      .pruned_infeasible = 0,
      .dropped = dropped,
      .iter_idx = 4,
      .iter_limit = 1000,
  });
  std::string const text = out.str();
  CHECK(text.find("Dropped viable regions: 1\n") != std::string::npos);
  CHECK(text.find("Dropped dominated regions: 1\n") != std::string::npos);
  CHECK(text.find("Dropped lb floor: -2\n") != std::string::npos);
}
