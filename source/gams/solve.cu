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
// The last line of a successful run is a tab-separated
//
//   RESULT<TAB>sense=min|max<TAB>primal=<value|none><TAB>dual=<value|none>
//
// in the file's own sense, which is what tools/minlp_status.py records into
// MINLP_STATUS.md. Everything else on stdout is progress narration, with one
// exception: a
//
//   PARAMS<TAB>shape=<name><TAB>partition_num=N<TAB>...
//
// line before the search, the machine-readable twin of the status line. It
// carries the *resolved* search shape -- after defaults, after the shape
// table, after auto-fitting -- because a bound is only reproducible from the
// values the run actually used, and three of the four have defaults that
// depend on the model or the device. minlp_status.py records it alongside the
// bound so MINLP_STATUS.md says how each number was obtained.
//
// The point of the frontend: no hand-written builder, no recompilation per
// instance. Compare source/morse_cluster_energy.cu, which builds the same
// ex8_6_2 problem by hand -- running both on ex8_6_2.gms should agree.
//
// No search-shape parameter is a template argument at this level any more.
// The positional shape argument now only picks *defaults*: --partition-num,
// --enumerate-cap, --sample-points and --max-cycle-size each override one.
//
// Three of those four defaults are the value the shape was originally tuned
// with. --max-cycle-size is not, and deliberately: it is the parameter that
// decides whether a run fits in device memory, the cost is knowable before
// anything is allocated, and the tabulated answer for a discrete model was
// 221 GiB. It is fitted to the device instead (resolve_max_cycle_size), and
// the status line marks it "(auto)" so a run's shape is still readable off
// its own output.
//
// The one value that still has to reach device codegen is the slot capacity
// (partition::SlotContext's array bound). It is handled by compiling a small
// ladder of capacities and dispatching to the narrowest rung that holds
// --max-cycle-size, rather than by baking one into the binary -- see
// capacity_ladder.hpp and design/RUNTIME_SHAPE.md §4.

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

#include "cuminlp/capacity_ladder.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/dag_print.hpp"
#include "cuminlp/gams.hpp"
#include "cuminlp/graph_driver.cuh"

namespace
{

/// A GraphDriver's final bounds plus the sample point that attained the
/// incumbent, kept alive past the driver's own lifetime. Both bounds are in
/// the *minimising* sense the driver works in; the sign flip for a Maximise
/// file happens where they are reported.
struct Solution {
  double bound;  ///< the incumbent, == upper
  double lower;  ///< dual bound: nothing beats this
  double upper;  ///< primal bound: best feasible value actually attained
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
      policy, iters, 1e-6, sample_points);
  double const bound = driver.solve(problem);
  auto const best = driver.best_point();
  return Solution{bound, driver.lower_bound(), driver.upper_bound(),
                  std::vector<double>(best.begin(), best.end())};
}

/**
 * @brief Read the hardware inputs the policy is allowed to see, once.
 *
 * Frozen here and never re-read: CompositionPolicy::choose must be a pure
 * function of (box, var_kinds, calibration), because materialise() re-invokes
 * it to decode a node's box and a differing answer would decode sidx against
 * the wrong radix. See design/RUNTIME_SHAPE.md §5.
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

auto probe_calibration(std::size_t max_cycle_size) -> cuminlp::SearchCalibration
{
  cuminlp::SearchCalibration calibration;
  calibration.max_cycle_size = max_cycle_size;

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

/// And for --max-cycle-size: the slot count each shape was tuned with. Only
/// a fallback now, for when the device budget cannot be read at all -- see
/// resolve_max_cycle_size. These numbers were preserved verbatim from the
/// hand-written drivers, and preserving them turned out to preserve a bug:
/// Discrete's 10 came from source/nvs09.cu at a revision where CYCLE_SIZE was
/// 10, a configuration that needs 221 GiB and has never run on a consumer
/// GPU. The shape whose defaults were "modelled on nvs09" could not solve
/// nvs09.
///
/// This is a *cap on the policy*, not the compiled capacity -- the ladder
/// rounds it up to the nearest instantiated rung, which costs registers and
/// nothing else (padding slots have fan-out 1).
auto fallback_max_cycle_size(Shape shape) -> std::size_t
{
  switch (shape) {
    case Shape::AllBinary: return 20;
    case Shape::Discrete:  return 10;
    case Shape::Mixed:     return 4;
  }
  return 4;
}

/// Fraction of free device memory auto-fitting will spend on the widest
/// composition. The remainder is headroom for the narrower compositions the
/// search also reaches: each is at least one slot's fan-out cheaper than the
/// last, so their total is a geometric tail over the widest, and a fan-out of
/// 2 -- the smallest a slot can have -- bounds that tail by the widest again.
/// Two thirds is comfortably inside that for any real fan-out while still
/// leaving the choice recognisably ambitious.
constexpr double auto_budget_fraction = 0.67;

/// Where --max-cycle-size comes from when it is not given.
struct ChosenCap {
  std::size_t value;
  bool automatic;  ///< false when the shape fallback had to be used
};

/**
 * @brief Pick the widest slot cap whose graphs fit on this device.
 *
 * The default used to be a per-shape constant, which meant the very first
 * command a user typed could fail with an out-of-memory report and a
 * suggestion to type a second one. The sizing information needed to get it
 * right is available before anything is allocated, so this does the search
 * the user would otherwise do by hand.
 *
 * Falls back to the shape's tuned constant only when cudaMemGetInfo gave
 * nothing to size against; guessing is still better than refusing to run.
 */
auto resolve_max_cycle_size(cuminlp::dag::Problem<double> const& problem,
                            cuminlp::FanOutSpec const& fan_out,
                            std::size_t sample_points,
                            std::size_t free_bytes,
                            Shape shape) -> ChosenCap
{
  if (free_bytes == 0) {
    return {fallback_max_cycle_size(shape), false};
  }
  auto const budget =
      static_cast<std::size_t>(static_cast<double>(free_bytes)
                               * auto_budget_fraction);
  std::size_t const fitted = cuminlp::dag::auto_max_cycle_size(
      problem, fan_out, sample_points, budget, cuminlp::max_capacity());
  // Zero means not even one slot fits. Run at 1 anyway: the build's own
  // out-of-memory report explains the per-element cost far better than
  // anything that could be said here, and it is the report the user needs.
  return {fitted == 0 ? std::size_t {1} : fitted, true};
}

/**
 * @brief Warn when --partition-num has quietly changed what integer slots do.
 *
 * enumerate_cap defaults to partition_num, so `--partition-num=2` on a model
 * of [3, 9] integers drops enumerate_cap to 2 as well and every integer slot
 * flips from IntegerEnumerate to IntegerBisect. Nothing about the status line
 * says so: it reports the flag that was set and the value it implied, both
 * looking exactly as asked for.
 *
 * States the mechanism and stops there, rather than predicting that the
 * search will suffer. Bisecting is cheaper per slot, so the auto-fitted cap
 * usually widens to compensate, and on nvs09 the bisecting run converges
 * while "fixing" it with --enumerate-cap=7 narrows the cap to 6 and stops
 * converging. The knob is worth knowing about; which way to turn it is not
 * something this can tell from here.
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
  auto usage = [&](std::ostream& out) {
    out << "usage: " << argv[0]
        << " [--dump-dag[=infix|nodes]] [--dump-only]"
        << " [--partition-num=N] [--enumerate-cap=N] [--sample-points=N]"
        << " [--max-cycle-size=N]"
        << " <model.gms> <iterations> [all-binary|discrete|mixed]\n";
  };

  // Printed on request to stdout at exit 0, never as part of an error. The
  // defaults are the questions this tool gets asked, so they are stated here
  // rather than left to USAGE.md.
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
           "  [shape]         all-binary|discrete|mixed; pins which defaults "
           "to use\n"
           "                  instead of inferring them from the model's "
           "variable kinds\n"
           "\n"
           "Search shape:\n"
           "  --partition-num=N   sub-intervals a continuous or "
           "bisected-integer slot\n"
           "                      splits into; >= 2. Default 2 (all-binary), "
           "7\n"
           "                      (discrete), 10 (mixed).\n"
           "  --enumerate-cap=N   largest integer domain still enumerated "
           "exactly rather\n"
           "                      than bisected. Defaults to --partition-num, "
           "which is\n"
           "                      a trap worth knowing about: lowering "
           "--partition-num\n"
           "                      lowers this too, and an integer whose domain "
           "no longer\n"
           "                      fits starts bisecting instead. Set it "
           "explicitly to\n"
           "                      decouple them.\n"
           "  --sample-points=N   points sampled per subdomain, for the "
           "incumbent.\n"
           "                      Default 1 (all-binary), 5 (discrete), 10 "
           "(mixed).\n"
           "  --max-cycle-size=N  variables one iteration acts on; <= 64. "
           "Default: the\n"
           "                      widest cap whose graphs fit in this device's "
           "free\n"
           "                      memory, shown on the status line as "
           "\"(auto)\".\n"
           "\n"
           "Inspection:\n"
           "  --dump-dag[=infix|nodes]  print the lowered Problem before "
           "solving\n"
           "  --dump-only               print and stop; needs no GPU\n"
           "  -h, --help                this message\n"
           "\n"
           "Exit codes: 0 solved or hit the iteration limit, 1 parse error, 2 "
           "bad flag\n"
           "value or rejected configuration, 3 out of device memory.\n";
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

  // --dump-only needs no iteration count; it never reaches the driver.
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

    // Unset fan-out and sampling flags fall back to the width this shape was
    // tuned with. enumerate_cap defaults to partition_num, as its
    // template-parameter ancestor did.
    std::size_t const chosen_partition_num =
        partition_num.value_or(default_partition_num(shape));
    cuminlp::FanOutSpec const fan_out {
        chosen_partition_num, enumerate_cap.value_or(chosen_partition_num)};

    std::size_t const chosen_sample_points =
        sample_points.value_or(default_sample_points(shape));

    // --max-cycle-size is the one default that is *measured* rather than
    // tabulated, because it is the one that decides whether the run fits in
    // memory at all, and the table's answer for a discrete model did not.
    ChosenCap cap {0, false};
    if (max_cycle_size) {
      cap = {*max_cycle_size, false};
    } else {
      cap = resolve_max_cycle_size(parsed.problem,
                                   fan_out,
                                   chosen_sample_points,
                                   free_device_bytes(),
                                   shape);
    }
    std::size_t const chosen_max_cycle_size = cap.value;

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
              << (cap.automatic ? " (auto, rung " : " (rung ") << rung
              << ")\n";

    // The same four numbers again, tab-separated, for tools/minlp_status.py.
    // Deliberately not the line above reformatted from a shared helper: this
    // one is a data format with a reader on the other end, and the values it
    // names are exactly the ones the flags of the same names set, so a
    // recorded run can be replayed by pasting them back. The rung is left
    // out for that reason -- it follows from max_cycle_size and there is no
    // flag that sets it -- and so is "(auto)", which says where the number
    // came from and not what it was.
    std::cout << "PARAMS\tshape=" << shape_name(shape)
              << "\tpartition_num=" << fan_out.partition_num()
              << "\tenumerate_cap=" << fan_out.enumerate_cap()
              << "\tsample_points=" << chosen_sample_points
              << "\tmax_cycle_size=" << chosen_max_cycle_size << '\n';

    if (partition_num && !enumerate_cap) {
      warn_on_implied_enumerate_cap(parsed.problem, fan_out);
    }

    Solution const solution = solve(parsed.problem,
                                    std::stoi(positional[1]),
                                    fan_out,
                                    chosen_sample_points,
                                    chosen_max_cycle_size);

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
  } catch (cuminlp::ResourceExhausted const& e) {
    // Its own exit code: unlike a bad flag value, the run was well-formed and
    // the same command may succeed on a larger GPU.
    std::cerr << "out of device memory: " << e.what() << '\n';
    return 3;
  }
}
