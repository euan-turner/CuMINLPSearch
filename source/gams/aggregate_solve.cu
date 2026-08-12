// Solve a GAMS scalar-format model with the aggregate-bounding backend.
//
//   aggregate_solve [flags] <model.gms> <iterations>
//
// design/AGGREGATE_BOUNDING.md. The device's parallel subdivision is used to
// *bound* k children rather than to enumerate N of them onto the host: the N
// per-subregion verdicts are reduced to k child bounds before anything
// crosses PCIe, and only those k children are enqueued.
//
// A separate binary from gams_solve, deliberately (§1, §10): the two backends
// are compared, not merged, and the parameter families have nothing in
// common. What is *not* separate is the output grammar -- the `PARAMS` and
// `RESULT` lines and ConsoleReporter's summary are byte-identical in shape to
// gams_solve's, so tools/minlp_status.py records both backends into one table
// with no changes to the scraper.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cuminlp/aggregate/cost.hpp"
#include "cuminlp/aggregate/driver.hpp"
#include "cuminlp/aggregate/policy.hpp"
#include "cuminlp/aggregate/selection.hpp"
#include "cuminlp/backend/aggregate/cache.cuh"
#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/config/problem_profile.hpp"
#include "cuminlp/gams.hpp"
#include "cuminlp/model/print.hpp"
#include "cuminlp/report/fan.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/report/telemetry.hpp"

namespace
{

auto free_device_bytes() -> std::size_t
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return 0;
  }
  return free_bytes;
}

/// The same fraction of free device memory config::resolve_shape's own fit
/// reserves, so a bound recorded here is against a comparable device budget.
constexpr double kBudgetFraction = 0.8;

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  auto usage = [&](std::ostream& out)
  {
    out << "usage: " << argv[0]
        << " [--branch-factor=k] [--partition-budget=N]"
           " [--selection=<rule>] [--sample-budget-fraction=F]"
           " [--sample-points=P] [--relative-width] [--tolerance=X]"
           " [--known-primal-bound=X] [--host-budget-bytes=N] [--verbose]"
           " [--dump-dag[=infix|nodes]] [--dump-only] [-h|--help]"
           " <model.gms> <iterations>\n";
  };

  auto help = [&]
  {
    usage(std::cout);
    std::cout
        << "\n"
           "Aggregate bounding (design/AGGREGATE_BOUNDING.md): the device "
           "evaluates N\n"
           "subregions per iteration and reduces them to k child bounds "
           "before anything\n"
           "crosses PCIe, so the pending frontier grows by k-1 per iteration "
           "rather\n"
           "than by up to N-1.\n"
           "\n"
           "Shape:\n"
           "  --branch-factor=k   children enqueued per iteration; a power of "
           "two,\n"
           "                      at least 2. Default 4.\n"
           "  --partition-budget=N  subregions evaluated per iteration; a "
           "power of two\n"
           "                      greater than k. 0 (the default) fits N to "
           "this\n"
           "                      device's free memory.\n"
           "  --sample-budget-fraction=F  share of the device budget given to "
           "the\n"
           "                      sampler. Default 0.10: a point element is "
           "about half\n"
           "                      an interval element, so this still buys "
           "~0.2*N points.\n"
           "  --sample-points=P   points drawn per sampler stratum. Default "
           "1.\n"
           "  --relative-width    the refine's greedy split compares each "
           "variable's\n"
           "                      live width against its own original width "
           "rather than\n"
           "                      the raw absolute width.\n"
           "\n"
           "Search:\n"
           "  --selection=<rule>  which pending node to explore next:\n"
           "                        least-lower-bound  (default) best-first "
           "on the\n"
           "                            dual bound -- what gams_solve does.\n"
           "                        reject-index      largest "
           "(incumbent - lb) /\n"
           "                            (hull_ub - lb); normalises out "
           "relaxation\n"
           "                            looseness. Re-keys on every "
           "incumbent.\n"
           "                        reject-index-stale  keys fixed at "
           "insertion. For\n"
           "                            measurement only.\n"
           "  --tolerance=X       primal/dual convergence tolerance. Default "
           "1e-6.\n"
           "  --known-primal-bound=X  seeds the incumbent; must be sound "
           "(upper bound\n"
           "                      for min, lower for max) or the search is "
           "wrong.\n"
           "\n"
           "Exit codes: 0 solved or hit the iteration limit, 1 parse error, "
           "2 bad flag\n"
           "value or rejected configuration, 3 out of device memory.\n";
  };

  std::size_t branch_factor = 4;
  std::size_t partition_budget = 0;
  std::size_t sample_points = 1;
  double sample_fraction = cuminlp::aggregate::default_sample_budget_fraction;
  double tolerance = 1e-6;
  auto rule = cuminlp::aggregate::SelectionRule::LeastLowerBound;
  std::string rule_name = "least-lower-bound";
  std::optional<double> known_primal_bound;
  std::size_t host_budget_bytes = 0;
  bool relative_width = false;
  bool verbose = false;
  bool dump = false;
  bool dump_only = false;
  auto style = cuminlp::model::PrintStyle::Infix;
  std::vector<std::string> positional;

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
  auto parse_double = [&](std::string const& value,
                          char const* flag) -> std::optional<double>
  {
    try {
      std::size_t consumed = 0;
      double const n = std::stod(value, &consumed);
      if (consumed != value.size()) {
        throw std::invalid_argument("trailing characters");
      }
      return n;
    } catch (std::exception const&) {
      std::cerr << flag << " expects a number, got '" << value << "'\n";
      return std::nullopt;
    }
  };

  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    auto value_of = [&](std::size_t prefix) { return arg.substr(prefix); };

    if (arg.rfind("--branch-factor=", 0) == 0) {
      auto v = parse_count(value_of(16), "--branch-factor");
      if (!v) {
        return 2;
      }
      branch_factor = *v;
    } else if (arg.rfind("--partition-budget=", 0) == 0) {
      auto v = parse_count(value_of(19), "--partition-budget");
      if (!v) {
        return 2;
      }
      partition_budget = *v;
    } else if (arg.rfind("--sample-points=", 0) == 0) {
      auto v = parse_count(value_of(16), "--sample-points");
      if (!v) {
        return 2;
      }
      sample_points = *v;
    } else if (arg.rfind("--sample-budget-fraction=", 0) == 0) {
      auto v = parse_double(value_of(25), "--sample-budget-fraction");
      if (!v) {
        return 2;
      }
      sample_fraction = *v;
    } else if (arg.rfind("--tolerance=", 0) == 0) {
      auto v = parse_double(value_of(12), "--tolerance");
      if (!v) {
        return 2;
      }
      tolerance = *v;
    } else if (arg.rfind("--known-primal-bound=", 0) == 0) {
      known_primal_bound = parse_double(value_of(21), "--known-primal-bound");
      if (!known_primal_bound) {
        return 2;
      }
    } else if (arg.rfind("--host-budget-bytes=", 0) == 0) {
      auto v = parse_count(value_of(20), "--host-budget-bytes");
      if (!v) {
        return 2;
      }
      host_budget_bytes = *v;
    } else if (arg.rfind("--selection=", 0) == 0) {
      rule_name = value_of(12);
      auto parsed = cuminlp::aggregate::parse_selection_rule(rule_name);
      if (!parsed) {
        std::cerr << "unknown selection rule '" << rule_name
                  << "'; expected least-lower-bound, reject-index or "
                     "reject-index-stale\n";
        return 2;
      }
      rule = *parsed;
    } else if (arg == "--relative-width") {
      relative_width = true;
    } else if (arg == "--verbose" || arg == "--telemetry") {
      verbose = true;
    } else if (arg == "--dump-dag" || arg.rfind("--dump-dag=", 0) == 0) {
      dump = true;
      if (arg.size() > 11) {
        std::string const v = arg.substr(11);
        if (v == "nodes") {
          style = cuminlp::model::PrintStyle::Nodes;
        } else if (v != "infix") {
          std::cerr << "unknown dump style '" << v
                    << "'; expected infix|nodes\n";
          return 2;
        }
      }
    } else if (arg == "--dump-only") {
      dump = true;
      dump_only = true;
    } else if (arg == "--help" || arg == "-h") {
      help();
      return 0;
    } else if (arg.rfind("--", 0) == 0) {
      std::cerr << "unknown option '" << arg << "'\n";
      usage(std::cerr);
      return 2;
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() < (dump_only ? 1u : 2u)) {
    usage(std::cerr);
    return 2;
  }

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
      cuminlp::model::PrintOptions print_options;
      print_options.style = style;
      print_options.var_names = parsed.var_names;
      std::cout << '\n';
      cuminlp::model::print_problem(std::cout, parsed.problem, print_options);
      if (parsed.sense == cuminlp::gams::Sense::Maximise) {
        std::cout << "\n  (the file maximises; the objective above has already "
                     "been negated for the minimising solver)\n";
      }
      std::cout << '\n';
      if (dump_only) {
        return 0;
      }
    }

    // Fit N and S to this device, exactly as --policy=bisection-budget fits B.
    std::size_t const n_buffers =
        cuminlp::backend::graph::buffer_node_count(parsed.problem);
    auto const device_budget = static_cast<std::size_t>(
        static_cast<double>(free_device_bytes()) * kBudgetFraction);
    auto const split =
        cuminlp::aggregate::split_device_budget(device_budget, sample_fraction);

    std::size_t const fitted = cuminlp::aggregate::fit_power_of_two(
        cuminlp::aggregate::bounder_element_bytes<double>(n_buffers),
        split.bounder_bytes);
    std::size_t const n = partition_budget != 0 ? partition_budget : fitted;
    std::size_t const sample_fan_out = std::max<std::size_t>(
        2,
        cuminlp::aggregate::fit_power_of_two(
            cuminlp::aggregate::sampler_element_bytes<double>(n_buffers)
                * std::max<std::size_t>(sample_points, 1),
            split.sampler_bytes));

    // WidestBranchPolicy validates k and N; a bad pair is a configuration
    // error rather than an assertion deep inside a graph build.
    auto policy =
        std::make_shared<const cuminlp::aggregate::WidestBranchPolicy<double>>(
            branch_factor,
            n,
            cuminlp::config::SearchCalibration {},
            relative_width);
    auto sampler_policy =
        std::make_shared<const cuminlp::aggregate::WidestBranchPolicy<double>>(
            branch_factor,
            std::max(sample_fan_out, branch_factor * 2),
            cuminlp::config::SearchCalibration {},
            relative_width);

    std::string const policy_name = std::string("aggregate/widest-branch")
        + (relative_width ? "-relative" : "") + "/" + rule_name;

    std::cout << "policy: " << policy_name
              << " (named), branch_factor: " << branch_factor
              << ", partition_budget: " << n
              << ", sample_fan_out: " << sampler_policy->total_fan_out()
              << ", sample_points: " << sample_points << "\n";
    std::cout << "PARAMS\tpolicy=" << policy_name << "\tsource=named"
              << "\tbranch_factor=" << branch_factor
              << "\tpartition_budget=" << n
              << "\tsample_fan_out=" << sampler_policy->total_fan_out()
              << "\tsample_points=" << sample_points
              << "\tsample_budget_fraction=" << sample_fraction
              << "\tselection=" << rule_name << '\n';

    auto fan = std::make_shared<cuminlp::report::ObserverFan>();
    fan->add(std::make_shared<cuminlp::report::ConsoleReporter>(
        cuminlp::config::profile_problem(parsed.problem), verbose));
    if (verbose) {
      fan->add(std::make_shared<cuminlp::report::TelemetryReporter>());
    }

    auto cache =
        std::make_shared<cuminlp::backend::aggregate::GraphRoleCache<double>>(
            parsed.problem,
            n,
            branch_factor,
            sampler_policy->total_fan_out(),
            sample_points,
            split.bounder_bytes,
            split.sampler_bytes,
            verbose);

    bool const maximise = parsed.sense == cuminlp::gams::Sense::Maximise;
    if (maximise && known_primal_bound) {
      known_primal_bound = std::optional(-*known_primal_bound);
    }

    cuminlp::aggregate::AggregateDriver<double> driver(
        policy,
        sampler_policy,
        cache,
        rule,
        static_cast<std::uint32_t>(std::stoul(positional[1])),
        tolerance,
        host_budget_bytes,
        fan,
        known_primal_bound);

    auto const outcome = driver.solve(parsed.problem);

    if (!outcome.selection_rule_engaged) {
      // §7.5: a reject-index run that never found an incumbent spent all of
      // itself on the lb fallback. Recording it as reject-index without
      // saying so would misattribute the result to the rule.
      std::cerr << "note: no incumbent was ever found, so --selection="
                << rule_name
                << " never left its lower-bound fallback; this run's ordering "
                   "was least-lower-bound throughout\n";
    }

    std::cout << "objective (" << (maximise ? "maximise" : "minimise") << "): "
              << (maximise ? -outcome.upper_bound : outcome.upper_bound)
              << '\n';

    double const primal = maximise ? -outcome.upper_bound : outcome.upper_bound;
    double const dual = maximise ? -outcome.lower_bound : outcome.lower_bound;
    bool const found_incumbent = outcome.found_incumbent();
    bool const found_dual =
        outcome.lower_bound > std::numeric_limits<double>::lowest();
    std::streamsize const prev = std::cout.precision(17);
    std::cout << "RESULT\tsense=" << (maximise ? "max" : "min") << "\tprimal=";
    if (found_incumbent) {
      std::cout << primal;
    } else {
      std::cout << "none";
    }
    std::cout << "\tdual=";
    if (found_dual) {
      std::cout << dual;
    } else {
      std::cout << "none";
    }
    std::cout << '\n';
    std::cout.precision(prev);

    if (outcome.best_point.size() == parsed.var_names.size()) {
      for (std::size_t i = 0; i < parsed.var_names.size(); ++i) {
        std::cout << "  " << parsed.var_names[i] << " = "
                  << outcome.best_point[i] << '\n';
      }
    } else {
      std::cout
          << "  (no feasible sample was found; no solution point to report)\n";
    }
    return 0;
  } catch (cuminlp::gams::ParseError const& e) {
    std::cerr << "parse error: " << e.what() << '\n';
    return 1;
  } catch (cuminlp::InvalidConfiguration const& e) {
    std::cerr << "configuration error: " << e.what() << '\n';
    return 2;
  } catch (cuminlp::ResourceExhausted const& e) {
    std::cerr << "out of device memory: " << e.what() << '\n';
    return 3;
  }
}
