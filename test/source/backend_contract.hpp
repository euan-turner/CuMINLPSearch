#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"

// The contract every backend must satisfy, written against
// cuminlp::backend's three role interfaces and RegionBackendFactory and
// against nothing else (design/MODULE_REFACTOR.md §12 stage 3, §13).
//
// This is the point of the abstraction: a second backend passes this file
// verbatim by handing `check_backend_contract` a different factory, and the
// properties below -- lattice-respecting draws, a witness that really attains
// the value it reports, a bound that really is below the exact minimum -- are
// the ones a swap must not silently break. They are stated here rather than
// per-backend precisely because no backend gets to define them.
namespace cuminlp::test
{

/**
 * @brief A problem plus an independent host-side reading of it.
 *
 * `objective` and `feasible` are recomputed on the host from a witness the
 * backend returns, so "the value it reported is the value at the point it
 * reported" is checked against something the backend had no part in.
 */
struct ContractModel
{
  const model::Problem<double>& problem;
  std::vector<cu::interval<double>> box;
  std::function<double(std::span<const double>)> objective;
  std::function<bool(std::span<const double>)> feasible;
};

namespace detail
{

/// Relative comparison: -Werror=float-equal is on, and a witness's objective
/// is recomputed in a different order on the host than on the device.
inline bool close(double a, double b, double tol = 1e-9)
{
  return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}

/// Invariant 19: the value a role reports is the objective at the point it
/// reports, so no sampler can make a reported bound unsound -- only weaker or
/// stronger. This is the whole licence for swapping samplers (§5.2).
inline void check_witness(const ContractModel& model,
                          const backend::CandidateResult<double>& result,
                          std::size_t n_elements)
{
  if (!result.found) {
    return;
  }
  REQUIRE(result.point.size() == model.box.size());
  CHECK(detail::close(result.value, model.objective(result.point)));
  CHECK(model.feasible(result.point));
  CHECK(result.index >= 0);
  CHECK(static_cast<std::size_t>(result.index) < n_elements);

  for (std::size_t v = 0; v < model.box.size(); ++v) {
    CHECK(result.point[v] >= model.box[v].lb);
    CHECK(result.point[v] <= model.box[v].ub);
  }
}

/// Invariant 6: Integer and Binary variables are drawn on the integer
/// lattice, cycled this iteration or not. The one property §5.2's uniform
/// sampler must reproduce exactly, and the one a JIT sampler must satisfy.
inline void check_on_lattice(const ContractModel& model,
                             std::span<const double> point)
{
  for (std::size_t v = 0; v < model.box.size(); ++v) {
    if (model.problem.var_kinds[v] == model::VarKind::Continuous) {
      continue;
    }
    CHECK(detail::close(point[v], std::round(point[v])));
  }
}

}  // namespace detail

/**
 * @brief Run the whole contract against one factory.
 *
 * `enumerable` must be a fully-enumerable assignment and `subdividing` one
 * that is not, so the bundle's optional enumerator is exercised both ways.
 * Both must be assignments the policy contract admits: pairwise distinct
 * var_ids naming live dimensions of `model.box` (§4.5). `model` must be
 * feasible at some point of `enumerable`'s grid, so that the witness checks
 * have a witness to check.
 */
inline void check_backend_contract(
    const backend::RegionBackendFactory<double>& factory,
    const ContractModel& model,
    const region::SlotAssignment& enumerable,
    const region::SlotAssignment& subdividing,
    const region::FanOutSpec& fan_out,
    std::size_t samples_per_region)
{
  REQUIRE(region::is_fully_enumerable(enumerable.composition));
  REQUIRE_FALSE(region::is_fully_enumerable(subdividing.composition));

  backend::BackendCapabilities const caps = factory.capabilities();
  backend::BuildBudget budget;
  budget.samples_per_region = samples_per_region;

  // ---- cost model: a price quoted before anything exists (§5.5) ----
  backend::RegionCostModel const cost = factory.cost_model(model.problem);
  CHECK(cost.bundle_bytes(100, samples_per_region, false) > 0);
  CHECK(cost.bundle_bytes(200, samples_per_region, false)
        >= cost.bundle_bytes(100, samples_per_region, false));
  CHECK(cost.bundle_bytes(100, samples_per_region, true)
        >= cost.bundle_bytes(100, samples_per_region, false));

  // ---- bundle shape: a missing role is a capability, not a surprise ----
  backend::SubdivisionBundle<double> flat =
      factory.build_subdivision(model.problem,
                                subdividing.composition,
                                fan_out,
                                budget,
                                backend::RoleRequest::all());
  REQUIRE(flat.bounder != nullptr);
  REQUIRE(flat.sampler != nullptr);
  CHECK(flat.enumerator == nullptr);  // not fully enumerable

  backend::SubdivisionBundle<double> exact =
      factory.build_subdivision(model.problem,
                                enumerable.composition,
                                fan_out,
                                budget,
                                backend::RoleRequest::all());
  REQUIRE(exact.bounder != nullptr);
  REQUIRE(exact.sampler != nullptr);
  CHECK((exact.enumerator != nullptr) == caps.exact_enumeration);

  // A role not asked for is not built. The driver reaches a Composition by
  // one path or the other, and a backend that builds both anyway charges the
  // device for graphs that iteration will not launch.
  backend::SubdivisionBundle<double> const bound_only =
      factory.build_subdivision(model.problem,
                                enumerable.composition,
                                fan_out,
                                budget,
                                backend::RoleRequest::subdividing());
  CHECK(bound_only.bounder != nullptr);
  CHECK(bound_only.sampler != nullptr);
  CHECK(bound_only.enumerator == nullptr);

  backend::SubdivisionBundle<double> const fathom_only =
      factory.build_subdivision(model.problem,
                                enumerable.composition,
                                fan_out,
                                budget,
                                backend::RoleRequest::enumerating());
  CHECK(fathom_only.bounder == nullptr);
  CHECK(fathom_only.sampler == nullptr);
  CHECK((fathom_only.enumerator != nullptr) == caps.exact_enumeration);

  // ---- bounder ----
  std::size_t const flat_regions =
      region::composition_fan_out(subdividing.composition, fan_out);
  CHECK(flat.bounder->n_regions() == flat_regions);

  backend::Region<double> const flat_region {model.box, subdividing};
  backend::BoundResult<double> const bounds = flat.bounder->bound(flat_region);
  REQUIRE(bounds.n_regions == flat_regions);
  REQUIRE(bounds.feasible.size() >= flat_regions);
  REQUIRE(bounds.obj_lb.size() >= flat_regions);
  for (std::size_t r = 0; r < flat_regions; ++r) {
    CHECK((bounds.feasible[r] == 0 || bounds.feasible[r] == 1));
  }

  // Bounding is a function of the region: two calls, one answer. Copied out
  // first, because the result is a view the second call invalidates.
  std::vector<double> const first_lb(
      bounds.obj_lb.begin(),
      bounds.obj_lb.begin() + static_cast<std::ptrdiff_t>(flat_regions));
  backend::BoundResult<double> const again = flat.bounder->bound(flat_region);
  for (std::size_t r = 0; r < flat_regions; ++r) {
    CHECK(detail::close(first_lb[r], again.obj_lb[r]));
  }

  // ---- enumerator, and the bound below it ----
  std::size_t const exact_regions =
      region::composition_fan_out(enumerable.composition, fan_out);
  backend::Region<double> const exact_region {model.box, enumerable};

  if (exact.enumerator != nullptr) {
    CHECK(exact.enumerator->n_points() == exact_regions);

    backend::CandidateResult<double> const found =
        exact.enumerator->enumerate(exact_region);
    // Not conditional: `enumerable` evaluates every point of its grid, so a
    // model whose feasible set contains one of them must yield a witness.
    // Callers owe the contract such a model -- without this the witness
    // checks below would pass by never running.
    REQUIRE(found.found);
    detail::check_witness(model, found, exact.enumerator->n_points());
    if (found.found) {
      detail::check_on_lattice(model, found.point);
    }

    // Exact evaluation is deterministic: it takes no salt, so a second call
    // on the same region is the same answer or the fathom branch is unsound.
    double const repeat = exact.enumerator->enumerate(exact_region).value;
    CHECK(detail::close(found.value, repeat));

    // Soundness across roles: the bounder's least lower bound over the same
    // region cannot exceed the exact minimum in it. This is the property
    // that makes pruning safe, and neither role can check it alone.
    if (found.found) {
      backend::BoundResult<double> const exact_bounds =
          exact.bounder->bound(exact_region);
      double least = std::numeric_limits<double>::max();
      for (std::size_t r = 0; r < exact_bounds.n_regions; ++r) {
        if (exact_bounds.feasible[r] != 0) {
          least = std::min(least, exact_bounds.obj_lb[r]);
        }
      }
      CHECK(least <= found.value + 1e-9);
    }
  }

  // ---- sampler ----
  CHECK(flat.sampler->n_samples() >= flat_regions);

  backend::CandidateResult<double> const drawn =
      flat.sampler->sample(flat_region, /*salt=*/7);
  // A sampler may legitimately miss a thin feasible set, so this is a CHECK
  // rather than a REQUIRE -- but on a model with a feasible set of positive
  // measure, a sampler that never lands in it is worth reporting.
  CHECK(drawn.found);
  detail::check_witness(model, drawn, flat.sampler->n_samples());
  if (drawn.found) {
    detail::check_on_lattice(model, drawn.point);
  }

  if (caps.deterministic_sampling) {
    std::vector<double> const witness(drawn.point.begin(), drawn.point.end());
    double const value = drawn.value;
    backend::CandidateResult<double> const redrawn =
        flat.sampler->sample(flat_region, /*salt=*/7);
    CHECK(detail::close(value, redrawn.value));
    CHECK(redrawn.point.size() == witness.size());
    for (std::size_t v = 0; v < witness.size(); ++v) {
      CHECK(detail::close(witness[v], redrawn.point[v]));
    }
  }

  // ---- over budget: facts, not advice (§5.6) ----
  backend::BuildBudget tiny;
  tiny.bytes = 1;
  tiny.samples_per_region = samples_per_region;
  try {
    factory.build_subdivision(model.problem,
                              enumerable.composition,
                              fan_out,
                              tiny,
                              backend::RoleRequest::all());
    FAIL("a one-byte budget must not admit a build");
  } catch (const backend::OverBudgetError& e) {
    backend::OverBudget const& facts = e.facts();
    CHECK(facts.budget_bytes == 1);
    CHECK(facts.needed_bytes > facts.budget_bytes);
    CHECK(facts.n_regions == exact_regions);
    CHECK(facts.composition == enumerable.composition);
    CHECK_FALSE(facts.role.empty());
    CHECK_FALSE(facts.element_breakdown.empty());
    CHECK_FALSE(facts.element_inventory.empty());
    // No advice in the payload: the widest cap that would fit is the
    // resolver's arithmetic, not the backend's.
    CHECK(std::string(e.what()).find("--max-cycle-size=") == std::string::npos);
  }
}

}  // namespace cuminlp::test
