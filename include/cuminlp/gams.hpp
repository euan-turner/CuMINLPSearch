#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cuminlp/model/problem.hpp"

namespace cuminlp::gams
{

/// Objective direction stated by the file's `Solve` statement.
enum class Sense
{
  Minimise,
  Maximise
};

struct ParseOptions
{
  /**
   * Box half-width substituted for a variable with no finite bound. The
   * interval evaluator returns useless bounds on an unbounded box and the
   * search driver would partition it forever, so free variables must be given
   * some box. Every substitution is reported as a warning.
   */
  double default_bound = 1e6;

  /**
   * Replace `x.fx = v` variables by the literal `v` before lowering, so they
   * become Const nodes (or fold away entirely) instead of degenerate [v, v]
   * search dimensions -- the same trick the hand-written symmetry breaking in
   * source/morse_cluster_energy.cu uses. Off => a real variable with a
   * degenerate box.
   */
  bool fold_fixed_to_const = true;

  /**
   * Reject any file declaring a Binary or Integer variable, the way this
   * frontend always did before the search driver gained integer branching.
   * Off by default now that it hasn't. Exists so `gams_report` can still A/B
   * a corpus run against that old baseline and attribute any coverage delta
   * to integrality specifically, rather than to anything else this change
   * touched.
   */
  bool reject_discrete = false;
};

/// A non-fatal observation about the parse. Never silent, never fatal.
struct Diagnostic
{
  int line;
  std::string message;
};

/// Why a file was rejected. Lets a benchmark harness skip what it must and
/// report what it skipped, without matching on message text.
enum class ErrorKind
{
  Syntax,  ///< malformed, or not a statement form we know
  NonScalar,  ///< sets/loops/tables: not GAMS scalar format
  Discrete,  ///< sos1/sos2/semicont/semiint: still no Problem form
  UnsupportedFunction,  ///< a real GAMS function we decline to approximate
  Unrepresentable,  ///< structurally fine, but has no form in Problem
  Io,  ///< the file could not be read
};

/// Thrown for anything that cannot be represented; never leaves a half-built
/// Problem behind.
class ParseError : public std::runtime_error
{
public:
  ParseError(int line,
             std::string const& message,
             ErrorKind kind = ErrorKind::Syntax);
  int line;
  ErrorKind kind;
};

/**
 * @brief A parsed GAMS scalar model plus the metadata a harness needs.
 *
 * @tparam T numerical precision
 */
template<typename T>
struct ParsedModel
{
  model::Problem<T> problem;

  /// Direction the file asked for. If Maximise, `problem`'s objective has
  /// ALREADY been negated (the solver only minimises); this field exists so a
  /// caller can flip the sign of the value it reports.
  Sense sense = Sense::Minimise;

  /// var_names[i] is the GAMS name of the variable at box_bounds[i].
  std::vector<std::string> var_names;
  /// constraint_names[i] names problem.constraints[i].
  std::vector<std::string> constraint_names;
  /// Starting point from `x.l =`, indexed like var_names. Empty if the file
  /// gave no levels.
  std::vector<T> initial_point;

  /// True iff the objective variable survived lowering as a real continuous
  /// search dimension -- neither the `=E=` definition pass nor the
  /// inequality-rewrite pass (frontend.cpp's eliminate_objective) could
  /// solve for it. Such a variable occupies a composition slot but is not a
  /// degree of freedom, which is why policy selection discounts it rather
  /// than treating it as evidence the problem needs a continuous search
  /// dimension.
  bool objvar_kept = false;

  std::vector<Diagnostic> warnings;
};

/// Parse GAMS scalar-format source text.
template<typename T>
auto parse(std::string_view source,
           ParseOptions const& options = {}) -> ParsedModel<T>;

/// Parse a GAMS scalar-format file.
template<typename T>
auto parse_file(std::filesystem::path const& path,
                ParseOptions const& options = {}) -> ParsedModel<T>;

extern template auto parse<double>(std::string_view,
                                   ParseOptions const&) -> ParsedModel<double>;
extern template auto parse<float>(std::string_view,
                                  ParseOptions const&) -> ParsedModel<float>;
extern template auto parse_file<double>(
    std::filesystem::path const&, ParseOptions const&) -> ParsedModel<double>;
extern template auto parse_file<float>(
    std::filesystem::path const&, ParseOptions const&) -> ParsedModel<float>;

}  // namespace cuminlp::gams
