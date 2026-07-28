// Solve a GAMS scalar-format model straight from its .gms file.
//
//   gams_solve [--dump-dag[=infix|nodes]] [--dump-only] \
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
// CycleSize/PartitionNum/SamplePoints are template parameters, but which .gms
// file to solve is a runtime choice, so a fixed set of shapes is
// pre-instantiated below and selected by the parsed model's variable kinds
// (see choose_shape). The optional third argument pins a shape explicitly,
// for a benchmark run that must not depend on the heuristic guessing right.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
 * @brief Construct a GreedyCompositionPolicy and the GraphDriver it drives
 * from the same (CycleSize, PartitionNum) and run the solve.
 *
 * GraphDriver's own doc comment notes that its EnumerateCap "must match the
 * policy's own", but nothing checks that -- TEST_EXTENSION.md records this as
 * an open, unimplemented runtime check. Building both from one set of
 * template parameters in this one function makes a mismatch unrepresentable
 * at this call site, without needing that general check to exist.
 */
template<std::size_t CycleSize, std::size_t PartitionNum, std::size_t SamplePoints>
auto solve_with(cuminlp::dag::Problem<double> const& problem, int iters) -> Solution
{
  auto policy = std::make_shared<
      cuminlp::GreedyCompositionPolicy<double, CycleSize, PartitionNum>>();
  cuminlp::GraphDriver<double, CycleSize, PartitionNum, SamplePoints> driver(
      policy, iters, 1e-9);
  double const bound = driver.solve(problem);
  auto const best = driver.best_point();
  return Solution{bound, std::vector<double>(best.begin(), best.end())};
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

auto solve(Shape shape, cuminlp::dag::Problem<double> const& problem, int iters)
    -> Solution
{
  switch (shape) {
    case Shape::AllBinary: return solve_with<20, 2, 1>(problem, iters);
    case Shape::Discrete:  return solve_with<10, 7, 5>(problem, iters);
    case Shape::Mixed:     return solve_with<4, 10, 10>(problem, iters);
  }
  return solve_with<4, 10, 10>(problem, iters);
}

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  auto usage = [&] {
    std::cerr << "usage: " << argv[0]
              << " [--dump-dag[=infix|nodes]] [--dump-only]"
              << " <model.gms> <iterations> [all-binary|discrete|mixed]\n";
  };

  bool dump = false;
  bool dump_only = false;
  auto style = cuminlp::dag::PrintStyle::Infix;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
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
    std::cout << "shape: " << shape_name(shape) << '\n';

    Solution const solution =
        solve(shape, parsed.problem, std::stoi(positional[1]));

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
  }
}
