// Solve a GAMS scalar-format model straight from its .gms file.
//
//   gams_solve [--dump-dag[=infix|nodes]] [--dump-only] \
//              [--partition-num=N] [--enumerate-cap=N] [--sample-points=N] \
//              [--max-cycle-size=N] \
//              <model.gms> <iterations> [all-binary|discrete|mixed]
//
// --dump-dag prints the lowered Problem before solving. Problem::validate()
// proves the DAG is well-formed, not that it is the *right* DAG: outside the
// two instances with a hand-built oracle in the test suite, reading the
// expression back is the only check there is. --dump-only stops after
// printing, which needs no GPU.
//
// The point of the frontend: no hand-written builder, no recompilation per
// instance. Compare source/morse_cluster_energy.cu, which builds the same
// ex8_6_2 problem by hand -- running both on ex8_6_2.gms should agree.
//
// No search-shape parameter is a template argument at this level any more.
// The positional shape argument now only picks *defaults*: --partition-num,
// --enumerate-cap, --sample-points and --max-cycle-size each override one,
// and every shape carries the value it was originally tuned with, so
// omitting every flag reproduces the old behaviour exactly.
//
// The one value that still has to reach device codegen is the slot capacity
// (partition::SlotContext's array bound). It is handled by compiling a small
// ladder of capacities and dispatching to the narrowest rung that holds
// --max-cycle-size, rather than by baking one into the binary -- see
// capacity_ladder.hpp and design/RUNTIME_SHAPE.md §4.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuminlp/capacity_ladder.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/dag_print.hpp"
#include "cuminlp/gams.hpp"
#include "cuminlp/graph_driver.cuh"

namespace
{

/// A GraphDriver's final bound plus the sample point that attained it, kept
/// alive past the driver's own lifetime.
struct Solution {
  double bound;
  std::vector<double> point;
};

/**
 * @brief Construct a GreedyCompositionPolicy and the GraphDriver it drives,
 * and run the solve.
 *
 * The mismatch this function used to guard against no longer exists: the
 * driver reads its fan-out widths off the policy it is given
 * (CompositionPolicy::fan_out), so there is only one copy of them to get
 * wrong. Previously each carried its own EnumerateCap template argument with
 * nothing checking they agreed, and constructing both here from one set of
 * template parameters was the workaround.
 */
template<std::size_t Capacity>
auto solve_with(cuminlp::dag::Problem<double> const& problem,
                int iters,
                cuminlp::FanOutSpec fan_out,
                std::size_t sample_points,
                cuminlp::SearchCalibration calibration) -> Solution
{
  auto policy =
      std::make_shared<cuminlp::GreedyCompositionPolicy<double, Capacity>>(
          fan_out, calibration);
  cuminlp::GraphDriver<double, Capacity> driver(
      policy, iters, 1e-9, sample_points);
  double const bound = driver.solve(problem);
  auto const best = driver.best_point();
  return Solution{bound, std::vector<double>(best.begin(), best.end())};
}

/**
 * @brief Read the hardware inputs the policy is allowed to see, once.
 *
 * Frozen here and never re-read: CompositionPolicy::choose must be a pure
 * function of (box, var_kinds, calibration), because materialise() re-invokes
 * it to decode a node's box and a differing answer would decode sidx against
 * the wrong radix. See design/RUNTIME_SHAPE.md §5.
 */
auto probe_calibration(std::size_t max_cycle_size) -> cuminlp::SearchCalibration
{
  cuminlp::SearchCalibration calibration;
  calibration.max_cycle_size = max_cycle_size;

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    calibration.free_device_bytes = free_bytes;
  }
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

enum class Shape { AllBinary, Discrete, Mixed };

auto shape_name(Shape shape) -> char const*
{
  switch (shape) {
    case Shape::AllBinary: return "all-binary";
    case Shape::Discrete:  return "discrete";
    case Shape::Mixed:     return "mixed";
  }
  return "?";
}

auto parse_shape(std::string const& s) -> std::optional<Shape>
{
  if (s == "all-binary") return Shape::AllBinary;
  if (s == "discrete") return Shape::Discrete;
  if (s == "mixed") return Shape::Mixed;
  return std::nullopt;
}

/**
 * @brief Pick a pre-instantiated shape from the parsed model's variable kinds.
 *
 * Each row below is tuned against a real hand-written driver, not invented:
 * all-binary matches source/autocorr_bern20_03.cu, discrete-no-continuous
 * matches source/nvs09.cu, and mixed/continuous matches this tool's own
 * original hardcoded shape (tuned for ex8_6_2). Growing this table needs a
 * measurement justifying the new row, not just a shape it doesn't yet cover.
 */
auto choose_shape(std::vector<cuminlp::dag::VarKind> const& kinds) -> Shape
{
  bool any_continuous = false;
  bool any_integer = false;
  for (auto kind : kinds) {
    if (kind == cuminlp::dag::VarKind::Continuous) any_continuous = true;
    if (kind == cuminlp::dag::VarKind::Integer) any_integer = true;
  }
  if (any_continuous) return Shape::Mixed;
  if (any_integer) return Shape::Discrete;
  return Shape::AllBinary;
}

/// The partition_num each shape was originally tuned with, used when
/// --partition-num is not given. Binary slots ignore it (their fan-out is
/// always 2), so all-binary's 2 is nominal.
auto default_partition_num(Shape shape) -> std::size_t
{
  switch (shape) {
    case Shape::AllBinary: return 2;
    case Shape::Discrete:  return 7;
    case Shape::Mixed:     return 10;
  }
  return 10;
}

/// Likewise for --sample-points: the count each shape was tuned with.
auto default_sample_points(Shape shape) -> std::size_t
{
  switch (shape) {
    case Shape::AllBinary: return 1;
    case Shape::Discrete:  return 5;
    case Shape::Mixed:     return 10;
  }
  return 10;
}

/// And for --max-cycle-size: the slot count each shape was tuned with. This
/// is a *cap on the policy*, not the compiled capacity -- the ladder rounds
/// it up to the nearest instantiated rung, which costs registers and nothing
/// else (padding slots have fan-out 1).
auto default_max_cycle_size(Shape shape) -> std::size_t
{
  switch (shape) {
    case Shape::AllBinary: return 20;
    case Shape::Discrete:  return 10;
    case Shape::Mixed:     return 4;
  }
  return 4;
}

auto solve(cuminlp::dag::Problem<double> const& problem,
           int iters,
           cuminlp::FanOutSpec fan_out,
           std::size_t sample_points,
           std::size_t max_cycle_size) -> Solution
{
  // One templated lambda, instantiated once per ladder rung; which rung runs
  // is a runtime decision. This is the only place the compile-time capacity
  // is chosen, and it replaces the old three-row switch over hardcoded
  // (CycleSize, PartitionNum, SamplePoints) triples.
  auto const calibration = probe_calibration(max_cycle_size);
  return cuminlp::dispatch_on_capacity(
      max_cycle_size,
      [&]<std::size_t Capacity>() -> Solution
      {
        return solve_with<Capacity>(
            problem, iters, fan_out, sample_points, calibration);
      });
}

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  auto usage = [&] {
    std::cerr << "usage: " << argv[0]
              << " [--dump-dag[=infix|nodes]] [--dump-only]"
              << " [--partition-num=N] [--enumerate-cap=N] [--sample-points=N]"
              << " [--max-cycle-size=N]"
              << " <model.gms> <iterations> [all-binary|discrete|mixed]\n";
  };

  bool dump = false;
  bool dump_only = false;
  auto style = cuminlp::dag::PrintStyle::Infix;
  std::optional<std::size_t> partition_num;
  std::optional<std::size_t> enumerate_cap;
  std::optional<std::size_t> sample_points;
  std::optional<std::size_t> max_cycle_size;
  std::vector<std::string> positional;

  // Shared by --partition-num/--enumerate-cap. Rejects anything std::stoull
  // wouldn't consume in full, so `--partition-num=8x` is an error rather
  // than a silent 8. Range checking beyond "is a number" is FanOutSpec's job.
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
    if (arg.rfind("--max-cycle-size=", 0) == 0) {
      max_cycle_size = parse_count(arg.substr(17), "--max-cycle-size");
      if (!max_cycle_size) return 2;
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
    if (arg.rfind("--", 0) == 0) {
      std::cerr << "unknown option '" << arg << "'\n";
      usage();
      return 2;
    }
    positional.push_back(arg);
  }

  // --dump-only needs no iteration count; it never reaches the driver.
  if (positional.size() < (dump_only ? 1u : 2u)) {
    usage();
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

    Shape shape = choose_shape(parsed.problem.var_kinds);
    if (positional.size() > 2) {
      auto override_shape = parse_shape(positional[2]);
      if (!override_shape) {
        std::cerr << "unknown shape '" << positional[2]
                  << "'; expected all-binary|discrete|mixed\n";
        return 2;
      }
      shape = *override_shape;
    }

    // Unset flags fall back to the width this shape was tuned with, so the
    // no-flag invocation is byte-for-byte the old behavior. enumerate_cap
    // defaults to partition_num, as its template-parameter ancestor did.
    std::size_t const chosen_partition_num =
        partition_num.value_or(default_partition_num(shape));
    cuminlp::FanOutSpec const fan_out {
        chosen_partition_num, enumerate_cap.value_or(chosen_partition_num)};

    std::size_t const chosen_sample_points =
        sample_points.value_or(default_sample_points(shape));
    std::size_t const chosen_max_cycle_size =
        max_cycle_size.value_or(default_max_cycle_size(shape));

    // Resolved before the status line rather than inside it: an
    // out-of-ladder cap throws, and doing that mid-`<<` would leave a
    // half-written line in front of the error message.
    std::size_t const rung = cuminlp::ladder_rung_for(chosen_max_cycle_size);

    // The rung is reported alongside the cap because they differ whenever
    // the cap isn't itself a rung: a cap of 20 rounds up to 32, which costs
    // registers but changes no result (padding slots have fan-out 1).
    std::cout << "shape: " << shape_name(shape)
              << ", partition_num: " << fan_out.partition_num()
              << ", enumerate_cap: " << fan_out.enumerate_cap()
              << ", sample_points: " << chosen_sample_points
              << ", max_cycle_size: " << chosen_max_cycle_size
              << " (rung " << rung << ")\n";

    Solution const solution = solve(parsed.problem,
                                    std::stoi(positional[1]),
                                    fan_out,
                                    chosen_sample_points,
                                    chosen_max_cycle_size);

    // The solver only minimises; a maximisation was negated on the way in.
    bool const maximise = parsed.sense == cuminlp::gams::Sense::Maximise;
    std::cout << "objective (" << (maximise ? "maximise" : "minimise") << "): "
              << (maximise ? -solution.bound : solution.bound) << '\n';
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
  } catch (cuminlp::ResourceExhausted const& e) {
    // Its own exit code: unlike a bad flag value, the run was well-formed and
    // the same command may succeed on a larger GPU.
    std::cerr << "out of device memory: " << e.what() << '\n';
    return 3;
  }
}
