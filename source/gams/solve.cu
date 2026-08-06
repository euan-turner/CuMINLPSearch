// Solve a GAMS scalar-format model straight from its .gms file.
//
//   gams_solve [--policy=<name>] [--list-policies] [--host-budget-bytes=<n>]
//              [--bounded-frontier]
//              [--dump-dag[=infix|nodes]] [--dump-only] [-h|--help]
//              <model.gms> <iterations>
//
// --dump-dag prints the lowered Problem before solving. The problem DAG
// is validated during parsing, but this just checks it is well-formed,
// not an exact semantic match to the original.
//
// The last line of a successful run is a tab-separated
//
//   RESULT<TAB>sense=min|max<TAB>primal=<value|none><TAB>dual=<value|none>
//
// for tools/minlp_status.py to record into MINLP_STATUS.md
// MINLP_STATUS.md. Everything else on stdout is progress narration, with one
// exception: a
//
//   PARAMS<TAB>policy=<name><TAB>source=auto|named|overridden<TAB>...
//
// line before the search, carrying the resolved search shape.
//
// With no --policy, the search shape comes entirely from the problem's
// variable kinds/counts and this device's free memory.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/config/run_spec.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/dag_print.hpp"
#include "cuminlp/gams.hpp"
#include "cuminlp/graph_replay.cuh"
#include "cuminlp/policy_catalogue.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/driver.hpp"

namespace
{


/**
 * @brief The result of a solve run in the minimising sense
 */
struct Solution {
  double bound;  ///< the incumbent, == upper
  double lower;  ///< dual bound: lower bound over all pending intervals
  double upper;  ///< primal bound: best feasible value actually attained
  std::vector<double> point;  ///< the sample point that attained `upper`
};

/**
  * @brief Construct a concrete CompositionPolicy and SearchDriver, then solve
  *
  * @param problem
  * @param spec the run's hyperparameters, fully resolved (design/
  *             MODULE_REFACTOR.md §7.1) -- `spec.calibration` is the one the
  *             policy is actually built against, distinct from the lighter
  *             calibration `config::resolve` fits the shape against
  * @return Solution
  */
auto solve_with(cuminlp::dag::Problem<double> const& problem,
                cuminlp::config::RunSpec const& spec) -> Solution
{
  std::shared_ptr<const cuminlp::CompositionPolicy<double>> policy =
      cuminlp::make_policy<double>(spec.policy_kind, spec.fan_out,
                                   spec.calibration);
  auto backend =
      std::make_shared<const cuminlp::dag::GraphBackendFactory<double>>();
  auto reporter = std::make_shared<cuminlp::report::ConsoleReporter>(
      cuminlp::profile_problem(problem));
  cuminlp::SearchDriver<double> driver(
      policy, backend, spec.iter_limit, spec.tolerance, spec.sample_points,
      spec.budgets.device_bytes, spec.budgets.host_bytes, spec.frontier,
      reporter);
  cuminlp::SolveOutcome<double> const outcome = driver.solve(problem);
  return Solution{outcome.upper_bound, outcome.lower_bound,
                  outcome.upper_bound, outcome.best_point};
}

/**
 * @brief Read the volume of free device memory available to the driver.
 *
 */
auto free_device_bytes() -> std::size_t
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return 0;
  }
  return free_bytes;
}

auto probe_calibration(std::size_t max_slots) -> cuminlp::SearchCalibration
{
  cuminlp::SearchCalibration calibration;
  calibration.max_cycle_size = max_slots;

  calibration.free_device_bytes = free_device_bytes();
  int device = 0;
  cudaDeviceProp props {};
  if (cudaGetDevice(&device) == cudaSuccess
      && cudaGetDeviceProperties(&props, device) == cudaSuccess)
  {
    calibration.multiprocessor_count =
        static_cast<std::size_t>(props.multiProcessorCount);
  }
  return calibration;
}

/**
 * @brief Warn when overriding --partition-num has quietly changed what
 *        integer slots do.
 *
 */
void warn_on_implied_enumerate_cap(cuminlp::dag::Problem<double> const& problem,
                                   cuminlp::FanOutSpec const& fan_out)
{
  std::size_t bisecting = 0;
  std::size_t largest = 0;
  for (std::size_t i = 0; i < problem.var_kinds.size(); ++i) {
    if (problem.var_kinds[i] != cuminlp::dag::VarKind::Integer) {
      continue;
    }
    auto const& b = problem.box_bounds[i];
    double const lo = std::ceil(b.lb);
    double const hi = std::floor(b.ub);
    if (hi < lo) {
      continue;
    }
    auto const domain = static_cast<std::size_t>(hi - lo) + 1;
    if (domain > fan_out.enumerate_cap()) {
      ++bisecting;
      largest = std::max(largest, domain);
    }
  }
  if (bisecting == 0) {
    return;
  }
  std::cout << "  note: --partition-num also set enumerate_cap to "
            << fan_out.enumerate_cap() << ", so " << bisecting
            << " integer variable(s) -- the widest of domain " << largest
            << " -- bisect\n"
            << "        instead of enumerating, and are never fathomed "
               "exactly. --enumerate-cap="
            << largest << "\n"
            << "        decouples the two if that was not intended.\n";
}

/**
 * @brief Fill in the policy-construction calibration a RunSpec doesn't carry
 *        on its own, then solve.
 *
 * `spec.max_slots` (already resolved, possibly overridden) is what the
 * built policy is actually capped at; probing it here rather than inside
 * `config::resolve` keeps that resolver a pure function of problem/device/
 * policy/overrides, with no device-property probe of its own (unchanged
 * from today's `probe_calibration` split).
 */
auto solve(cuminlp::dag::Problem<double> const& problem,
           cuminlp::config::RunSpec spec) -> Solution
{
  spec.calibration = probe_calibration(spec.max_slots);
  return solve_with(problem, spec);
}

/// One line per roster row: name, rules, evidence, provisional marker.
/// Needs no GPU and no model -- it is pure data (policy_catalogue.hpp's
/// policy_roster). Shared by --list-policies and an unknown --policy name's
/// error path, which exits 2 with the roster listed.
void print_roster(std::ostream& out)
{
  auto describe_partition = [](cuminlp::PartitionRule const& r) -> std::string
  {
    if (r.mode == cuminlp::PartitionRule::Mode::Pin) {
      return "pin(" + std::to_string(r.pinned) + ")";
    }
    return "fit";
  };
  auto describe_enumerate = [](cuminlp::EnumerateRule const& r) -> std::string
  {
    switch (r.mode) {
      case cuminlp::EnumerateRule::Mode::CoverDomains:
        return "cover-domains(" + std::to_string(r.ceiling) + ")";
      case cuminlp::EnumerateRule::Mode::FollowPartition:
        return "follow-partition";
      case cuminlp::EnumerateRule::Mode::Pin:
        return "pin(" + std::to_string(r.pinned) + ")";
    }
    return "?";
  };
  auto describe_cycle = [](cuminlp::CycleRule const& r) -> std::string
  {
    if (r.mode == cuminlp::CycleRule::Mode::Pin) {
      return "pin(" + std::to_string(r.pinned) + ")";
    }
    return "fit (fallback " + std::to_string(r.pinned) + ")";
  };

  for (cuminlp::PolicyProfile const& p : cuminlp::policy_roster) {
    out << "  " << p.name << "\n"
        << "      partition=" << describe_partition(p.partition)
        << " enumerate=" << describe_enumerate(p.enumerate)
        << " sample_points=" << p.sample_points
        << " cycle=" << describe_cycle(p.cycle) << "\n"
        << "      evidence: " << p.evidence
        << (p.provisional ? " (provisional -- no measurement behind it yet)"
                          : "")
        << "\n";
  }
}

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  auto usage = [&](std::ostream& out) {
    out << "usage: " << argv[0]
        << " [--policy=<name>] [--list-policies]"
        << " [--host-budget-bytes=<n>] [--bounded-frontier]"
        << " [--dump-dag[=infix|nodes]] [--dump-only] [-h|--help]"
        << " <model.gms> <iterations>\n";
  };

  // Printed on request to stdout at exit 0, never as part of an error.
  auto help = [&] {
    usage(std::cout);
    // Adjacent literals rather than a raw string: nvcc's preprocessor does
    // not accept R"(...)" here.
    std::cout
        << "\n"
           "Solves a GAMS scalar-format model straight from its .gms file.\n"
           "\n"
           "Positional:\n"
           "  <model.gms>     model to solve\n"
           "  <iterations>    branch-and-bound iteration limit (omit with "
           "--dump-only)\n"
           "\n"
           "Policy:\n"
           "  --policy=<name>   use this named composition policy instead of "
           "selecting\n"
           "                    one automatically from the model's variable "
           "kinds and\n"
           "                    this device's free memory. See "
           "--list-policies.\n"
           "  --list-policies   print the policy roster (name, rules, "
           "evidence,\n"
           "                    provisional status) and exit; needs no GPU.\n"
           "\n"
           "Memory:\n"
           "  --host-budget-bytes=N  cap the host memory the search's pending\n"
           "                      frontier and interval history hold together."
           "\n"
           "                      Reaching it ends the run with the bracket\n"
           "                      intact rather than in the OOM killer; "
           "nothing\n"
           "                      is discarded, so the bounds are the ones the"
           "\n"
           "                      search had already earned. 0 (the default)\n"
           "                      measures half of MemAvailable.\n"
           "  --bounded-frontier  on reaching the budget, discard the "
           "worst-bounded\n"
           "                      pending regions and keep searching instead "
           "of\n"
           "                      stopping. Off by default: it changes which\n"
           "                      regions the search explores. The reported "
           "bounds\n"
           "                      stay sound and say what was discarded (see\n"
           "                      `Dropped ...` in the summary).\n"
           "\n"
           "Inspection:\n"
           "  --dump-dag[=infix|nodes]  print the lowered Problem before "
           "solving\n"
           "  --dump-only               print and stop; needs no GPU\n"
           "  -h, --help                this message\n"
           "\n"
           "Experiments (research instrumentation, not configuration -- see "
           "USAGE.md):\n"
           "  --partition-num=N   overrides the resolved partition_num; >= "
           "2.\n"
           "  --enumerate-cap=N   overrides the resolved enumerate_cap; >= "
           "1. Given\n"
           "                      --partition-num without this, "
           "enumerate_cap follows\n"
           "                      it instead of staying at what the policy "
           "resolved.\n"
           "  --sample-points=N   overrides the resolved sample_points.\n"
           "  --max-slots=N       overrides the resolved max_slots; <= 64. "
           "--max-cycle-size=N\n"
           "                      is a deprecated alias for this, kept for "
           "one release.\n"
           "  A run using any of these reports source=overridden instead of "
           "auto/named.\n"
           "\n"
           "Exit codes: 0 solved or hit the iteration limit (and --help, "
           "--list-policies),\n"
           "1 parse error, 2 bad flag value or rejected configuration "
           "(including an\n"
           "unknown --policy name, or one that doesn't apply to this "
           "model's variable\n"
           "kinds), 3 out of device memory.\n";
  };

  bool dump = false;
  bool dump_only = false;
  bool list_policies = false;
  auto style = cuminlp::dag::PrintStyle::Infix;
  std::optional<std::string> policy_name;
  std::optional<std::size_t> partition_num;
  std::optional<std::size_t> enumerate_cap;
  std::optional<std::size_t> sample_points;
  std::optional<std::size_t> max_slots;
  std::size_t host_budget_bytes = 0;
  bool bounded_frontier = false;
  std::vector<std::string> positional;

  // Shared by --partition-num/--enumerate-cap/--sample-points/--max-slots/
  // --max-cycle-size. Rejects anything std::stoull wouldn't consume in full,
  // so `--partition-num=8x` is an error rather than a silent 8. Range
  // checking beyond "is a number" is FanOutSpec's/CompositionPolicy's job.
  auto parse_count = [&](std::string const& value,
                         char const* flag) -> std::optional<std::size_t>
  {
    try {
      std::size_t consumed = 0;
      unsigned long long const n = std::stoull(value, &consumed);
      if (consumed != value.size()) {
        throw std::invalid_argument("trailing characters");
      }
      return static_cast<std::size_t>(n);
    } catch (std::exception const&) {
      std::cerr << flag << " expects a non-negative integer, got '" << value
                << "'\n";
      return std::nullopt;
    }
  };

  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    if (arg.rfind("--policy=", 0) == 0) {
      policy_name = arg.substr(9);
      continue;
    }
    if (arg == "--list-policies") {
      list_policies = true;
      continue;
    }
    if (arg.rfind("--partition-num=", 0) == 0) {
      partition_num = parse_count(arg.substr(16), "--partition-num");
      if (!partition_num) return 2;
      continue;
    }
    if (arg.rfind("--enumerate-cap=", 0) == 0) {
      enumerate_cap = parse_count(arg.substr(16), "--enumerate-cap");
      if (!enumerate_cap) return 2;
      continue;
    }
    if (arg.rfind("--sample-points=", 0) == 0) {
      sample_points = parse_count(arg.substr(16), "--sample-points");
      if (!sample_points) return 2;
      continue;
    }
    if (arg.rfind("--max-slots=", 0) == 0) {
      max_slots = parse_count(arg.substr(12), "--max-slots");
      if (!max_slots) return 2;
      continue;
    }
    if (arg.rfind("--max-cycle-size=", 0) == 0) {
      // Deprecated alias for --max-slots (design/MODULE_REFACTOR.md §7.4).
      // To stderr, not stdout: stdout is the data channel
      // tools/minlp_status.py scrapes, and this notice isn't part of it.
      std::cerr << "note: --max-cycle-size is deprecated; use --max-slots "
                   "instead\n";
      max_slots = parse_count(arg.substr(17), "--max-cycle-size");
      if (!max_slots) return 2;
      continue;
    }
    if (arg.rfind("--host-budget-bytes=", 0) == 0) {
      auto const parsed_budget =
          parse_count(arg.substr(20), "--host-budget-bytes");
      if (!parsed_budget) return 2;
      host_budget_bytes = *parsed_budget;
      continue;
    }
    if (arg == "--bounded-frontier") {
      bounded_frontier = true;
      continue;
    }
    if (arg == "--dump-dag" || arg.rfind("--dump-dag=", 0) == 0) {
      dump = true;
      if (arg.size() > 11) {
        std::string const value = arg.substr(11);
        if (value == "nodes") {
          style = cuminlp::dag::PrintStyle::Nodes;
        } else if (value == "infix") {
          style = cuminlp::dag::PrintStyle::Infix;
        } else {
          std::cerr << "unknown dump style '" << value
                    << "'; expected infix|nodes\n";
          return 2;
        }
      }
      continue;
    }
    if (arg == "--dump-only") {
      // Implies --dump-dag: asking only for the dump and getting nothing
      // would be a pure footgun.
      dump = true;
      dump_only = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      help();
      return 0;
    }
    if (arg.rfind("--", 0) == 0) {
      std::cerr << "unknown option '" << arg << "'\n";
      usage(std::cerr);
      return 2;
    }
    positional.push_back(arg);
  }

  if (list_policies) {
    print_roster(std::cout);
    return 0;
  }

  // The positional shape argument (all-binary|discrete|mixed) is gone; a
  // third positional is a specific, common mistake worth naming rather than
  // a bare usage dump (design/POLICY_SELECTION.md).
  if (positional.size() > 2) {
    std::cerr << "'" << positional[2]
              << "' is a third positional argument, which is no longer a "
                 "shape name; use --policy=<name> instead (see "
                 "--list-policies)\n";
    return 2;
  }

  // --dump-only needs no iteration count; it never reaches the driver.
  if (positional.size() < (dump_only ? 1u : 2u)) {
    usage(std::cerr);
    return 2;
  }

  // Hoisted out of the try only so the over-budget handler below can cost its
  // advice against this model; empty if the failure predates profiling.
  std::optional<cuminlp::ProblemProfile> profile_for_report;

  try {
    auto parsed = cuminlp::gams::parse_file<double>(positional[0]);

    std::cout << positional[0] << ": " << parsed.var_names.size()
              << " variables, " << parsed.problem.constraints.size()
              << " constraints, " << parsed.problem.graph.nodes.size()
              << " DAG nodes\n";
    for (auto const& w : parsed.warnings) {
      std::cout << "  warning (line " << w.line << "): " << w.message << '\n';
    }

    if (dump) {
      cuminlp::dag::PrintOptions print_options;
      print_options.style = style;
      print_options.var_names = parsed.var_names;
      std::cout << '\n';
      cuminlp::dag::print_problem(std::cout, parsed.problem, print_options);
      // The frontend negates a Maximise objective on the way in, so what is
      // printed above is what the solver minimises, not what the file wrote.
      if (parsed.sense == cuminlp::gams::Sense::Maximise) {
        std::cout << "\n  (the file maximises; the objective above has already "
                     "been negated for the minimising solver)\n";
      }
      std::cout << '\n';
      if (dump_only) return 0;
    }

    cuminlp::ProblemProfile problem_profile =
        cuminlp::profile_problem(parsed.problem);
    problem_profile.objvar_kept = parsed.objvar_kept;
    profile_for_report = problem_profile;

    cuminlp::SearchCalibration selection_calibration;
    selection_calibration.free_device_bytes = free_device_bytes();

    cuminlp::PolicyProfile policy {};
    cuminlp::config::Provenance base_source;
    if (policy_name) {
      auto found = cuminlp::lookup_policy(*policy_name);
      if (!found) {
        std::cerr << "unknown policy '" << *policy_name << "'; available "
                     "policies:\n";
        print_roster(std::cerr);
        return 2;
      }
      policy = *found;
      if (!cuminlp::is_applicable(policy, problem_profile)) {
        std::cerr << "policy '" << *policy_name
                  << "' is not applicable to this model: "
                  << problem_profile.num_binary << " binary, "
                  << problem_profile.num_integer << " integer, "
                  << problem_profile.num_continuous << " continuous "
                  << "variable(s)"
                  << (problem_profile.objvar_kept
                          ? " (one continuous is the kept objective variable)"
                          : "")
                  << "; pick a policy whose rules match these kinds, or omit "
                     "--policy to select one automatically. See "
                     "--list-policies.\n";
        return 2;
      }
      base_source = cuminlp::config::Provenance::Named;
    } else {
      policy = cuminlp::select_policy(problem_profile, selection_calibration);
      base_source = cuminlp::config::Provenance::Auto;
    }

    cuminlp::config::OverrideSet overrides;
    overrides.partition_num = partition_num;
    overrides.enumerate_cap = enumerate_cap;
    overrides.sample_points = sample_points;
    overrides.max_slots = max_slots;

    // The backend's four cost coefficients, so the fit can be run before any
    // graph exists (design/MODULE_REFACTOR.md §5.5).
    cuminlp::config::RunSpec spec = cuminlp::config::resolve(
        policy,
        problem_profile,
        selection_calibration,
        cuminlp::backend::graph::cost_model_for<double>(parsed.problem),
        base_source,
        overrides);
    spec.budgets.host_bytes = host_budget_bytes;
    spec.frontier = bounded_frontier ? cuminlp::FrontierPolicy::Compact
                                     : cuminlp::FrontierPolicy::StopAtBudget;
    spec.iter_limit = static_cast<std::uint32_t>(std::stoi(positional[1]));

    std::cout << "policy: " << spec.policy_name << " ("
              << cuminlp::config::to_string(spec.source)
              << "), partition_num: " << spec.fan_out.partition_num()
              << ", enumerate_cap: " << spec.fan_out.enumerate_cap()
              << ", sample_points: " << spec.sample_points
              << ", max_slots: " << spec.max_slots << "\n";

    // The machine-readable twin, for tools/minlp_status.py. `overrides=`
    // only appears on the overridden path, and lists only the flags actually
    // given.
    std::cout << "PARAMS\tpolicy=" << spec.policy_name
              << "\tsource=" << cuminlp::config::to_string(spec.source)
              << "\tpartition_num=" << spec.fan_out.partition_num()
              << "\tenumerate_cap=" << spec.fan_out.enumerate_cap()
              << "\tsample_points=" << spec.sample_points
              << "\tmax_slots=" << spec.max_slots;
    if (overrides.any()) {
      std::cout << "\toverrides=";
      bool first = true;
      auto emit_override = [&](char const* name,
                               std::optional<std::size_t> const& value)
      {
        if (!value) return;
        if (!first) std::cout << ",";
        std::cout << name << "=" << *value;
        first = false;
      };
      emit_override("partition_num", overrides.partition_num);
      emit_override("enumerate_cap", overrides.enumerate_cap);
      emit_override("sample_points", overrides.sample_points);
      emit_override("max_slots", overrides.max_slots);
    }
    std::cout << '\n';

    if (partition_num && !enumerate_cap) {
      warn_on_implied_enumerate_cap(parsed.problem, spec.fan_out);
    }

    Solution const solution = solve(parsed.problem, spec);

    // The solver only minimises; a maximisation was negated on the way in.
    bool const maximise = parsed.sense == cuminlp::gams::Sense::Maximise;
    std::cout << "objective (" << (maximise ? "maximise" : "minimise") << "): "
              << (maximise ? -solution.bound : solution.bound) << '\n';

    // One grep-able line carrying both bounds in the file's own sense, for
    // MINLP_STATUS.md's record step. Negating swaps which is which: the
    // incumbent of min(-f) is the best attained value of max(f), so it stays
    // the primal bound, and the driver's lower bound stays the dual one.
    double const primal = maximise ? -solution.upper : solution.upper;
    double const dual = maximise ? -solution.lower : solution.lower;
    bool const found_incumbent =
        solution.upper < std::numeric_limits<double>::max();
    std::streamsize const prev_precision = std::cout.precision(17);

    // A dual bound still at its initial sentinel means the search never
    // dequeued anything, so there is no bound to record -- distinct from one
    // that is merely weak, and the tracker must not conflate them.
    bool const found_dual = solution.lower > std::numeric_limits<double>::lowest();
    std::cout << "RESULT\tsense=" << (maximise ? "max" : "min") << "\tprimal=";
    if (found_incumbent) std::cout << primal;
    else std::cout << "none";
    std::cout << "\tdual=";
    if (found_dual) std::cout << dual;
    else std::cout << "none";
    std::cout << '\n';
    std::cout.precision(prev_precision);
    if (solution.point.size() == parsed.var_names.size()) {
      for (std::size_t i = 0; i < parsed.var_names.size(); ++i) {
        std::cout << "  " << parsed.var_names[i] << " = " << solution.point[i] << '\n';
      }
    } else {
      std::cout << "  (no feasible sample was found; no solution point to report)\n";
    }
    return 0;
  } catch (cuminlp::gams::ParseError const& e) {
    std::cerr << "parse error: " << e.what() << '\n';
    return 1;
  } catch (cuminlp::InvalidConfiguration const& e) {
    std::cerr << "configuration error: " << e.what() << '\n';
    return 2;
  } catch (cuminlp::backend::OverBudgetError const& e) {
    // Its own exit code: unlike a bad flag value, the run was well-formed and
    // the same command may succeed on a larger GPU. The backend threw the
    // facts; the report is written here, where the problem profile the advice
    // is costed against is in hand (design/MODULE_REFACTOR.md §5.6).
    std::cerr << "out of device memory: "
              << (profile_for_report
                      ? cuminlp::explain_over_budget(e.facts(),
                                                     *profile_for_report)
                      : std::string(e.what()))
              << '\n';
    return 3;
  } catch (cuminlp::ResourceExhausted const& e) {
    std::cerr << "out of device memory: " << e.what() << '\n';
    return 3;
  }
}
