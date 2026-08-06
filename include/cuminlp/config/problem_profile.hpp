#pragma once

#include <algorithm>
#include <cstddef>

#include "cuminlp/model/dag.hpp"
#include "cuminlp/model/problem.hpp"

namespace cuminlp::config
{

/// The problem-side facts the resolver and classifier need, and nothing
/// else: a pure function of this struct, a SearchCalibration and a
/// backend::RegionCostModel must reproduce a resolved shape.
///
/// It holds nothing backend-shaped. `buffer_nodes` used to live here and was
/// the one field that described an evaluator rather than a model; the
/// resolver now gets that through the cost model instead
/// (design/MODULE_REFACTOR.md §5.5), which is what lets profile_problem()
/// depend on nothing but the Problem.
struct ProblemProfile
{
  std::size_t num_binary = 0;
  std::size_t num_integer = 0;
  std::size_t num_continuous = 0;
  std::size_t largest_integer_domain = 0;  ///< over the root box, 0 if none

  /// True iff the objective variable survived lowering as a real continuous
  /// variable (neither the `=E=` nor the inequality elimination pass could
  /// solve for it).
  bool objvar_kept = false;
};

/**
 * @brief Derive a ProblemProfile from a parsed problem's variable kinds and
 *        bounds.
 *
 * Does not (and cannot) set `objvar_kept`: that fact comes from how the GAMS
 * frontend lowered the objective, not from anything a bare `model::Problem<T>`
 * records. Callers with a `gams::ParsedModel<T>` set it separately.
 */
template<typename T>
ProblemProfile profile_problem(const model::Problem<T>& problem)
{
  ProblemProfile out;
  for (std::size_t i = 0; i < problem.var_kinds.size(); ++i) {
    switch (problem.var_kinds[i]) {
      case model::VarKind::Binary:
        ++out.num_binary;
        break;
      case model::VarKind::Integer: {
        ++out.num_integer;
        // Number of integers in [ceil(lb), floor(ub)] -- same snapping as
        // GreedyCompositionPolicy::integer_domain_size, and 0 (not an
        // underflowed huge size_t) when the box has no integer point at all.
        T const lo = std::ceil(problem.box_bounds[i].lb);
        T const hi = std::floor(problem.box_bounds[i].ub);
        std::size_t const domain =
            hi < lo ? 0 : static_cast<std::size_t>(hi - lo) + 1;
        out.largest_integer_domain =
            std::max(out.largest_integer_domain, domain);
        break;
      }
      case model::VarKind::Continuous:
        ++out.num_continuous;
        break;
    }
  }
  return out;
}

}  // namespace cuminlp::config
