#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include <cuinterval/interval.h>

#include "cuminlp/backend/cost_model.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/format.hpp"

// The three roles a backend plays, named for what they do rather than for how
// they are built (design/MODULE_REFACTOR.md §5.1).
//
// The driver asks a backend for exactly three things -- candidates, bounds and
// exact evaluation -- and today gets all three from instantiations of one
// CUDA-graph class. Naming the roles is the whole of the abstraction: it lets
// the driver's call sites read as what they mean, lets a backend family
// declare a missing role as a capability rather than as a null pointer the
// caller has to guess about, and lets the roles have different shapes and
// lifetimes.
//
// This header is host-only by construction -- it is what `search` is allowed
// to name, and `search` must compile without a CUDA toolchain (§3.1).
namespace cuminlp::backend
{

/// One parent box plus the slots acting on it.
///
/// The sampler will take a bare box once §5.2 decouples it from the
/// composition; until then all three roles subdivide, so all three take this.
template<typename T>
struct Region
{
  std::span<const cu::interval<T>> box;  ///< n_vars entries
  const SlotAssignment& assignment;
};

/**
 * @brief One evaluated witness: the best feasible point a role found.
 *
 * A view over storage the role instance owns, valid until the next call on
 * that instance.
 *
 * `point` is the witness for `value`, indexed by variable. It is all-NaN when
 * `found` is false -- rather than empty, as §5.1 sketches -- because that is
 * the existing contract of the CUDA-graph backend's candidate buffer and
 * stage 3 is behaviour-preserving. Check `found` before reading it.
 */
template<typename T>
struct CandidateResult
{
  bool found = false;
  T value {};  ///< objective at the witness
  std::span<const T> point;
  std::int64_t index = -1;  ///< flat element index, for diagnostics
};

/// Sound per-subregion bounds and feasibility. A view over storage the role
/// instance owns, valid until the next call on that instance.
template<typename T>
struct BoundResult
{
  std::span<const unsigned char> feasible;  ///< per subregion, 1 = not excluded
  std::span<const T> obj_lb;  ///< per subregion
  std::size_t n_regions = 0;
};

/// Draws feasible candidates from a region; the source of incumbents.
///
/// `salt` is the driver's iteration index, so a revisited box draws fresh
/// points. A backend that satisfies `deterministic_sampling` must draw the
/// same points for the same (region, salt).
template<typename T>
class RegionSampler
{
public:
  virtual ~RegionSampler() = default;

  virtual CandidateResult<T> sample(const Region<T>& region,
                                    std::uint64_t salt) = 0;
  virtual std::size_t n_samples() const = 0;
};

/// Sound per-subregion bounds and feasibility; the source of pruning.
template<typename T>
class RegionBounder
{
public:
  virtual ~RegionBounder() = default;

  virtual BoundResult<T> bound(const Region<T>& region) = 0;
  virtual std::size_t n_regions() const = 0;
};

/// Exact evaluation of a fully-enumerable region; the source of fathoming.
///
/// The value it returns is the true minimum over every point the region
/// enumerates, not a bound -- which is what licenses the driver to fathom
/// without enqueueing children.
template<typename T>
class RegionEnumerator
{
public:
  virtual ~RegionEnumerator() = default;

  virtual CandidateResult<T> enumerate(const Region<T>& region) = 0;
  virtual std::size_t n_points() const = 0;
};

/// What a backend family can and cannot do. A missing role is a capability,
/// not a null pointer the driver has to interpret.
struct BackendCapabilities
{
  bool exact_enumeration = true;  ///< can supply a RegionEnumerator
  bool deterministic_sampling = true;  ///< same region + salt => same points
};

/**
 * @brief The roles one Composition needs, budgeted and built together.
 *
 * `enumerator` is null when the composition is not fully enumerable, or when
 * `capabilities().exact_enumeration` is false -- in which case the driver
 * never takes the fathom branch and enqueues children instead. That is a
 * capability difference which changes performance, not answers.
 *
 * `sampler` is in the bundle only while it is still composition-coupled.
 * §5.2 makes it a function of the problem alone, built once per solve, at
 * which point it leaves this struct for a `build_sampler` entry point of its
 * own and stops being charged per composition.
 */
template<typename T>
struct SubdivisionBundle
{
  std::unique_ptr<RegionSampler<T>> sampler;
  std::unique_ptr<RegionBounder<T>> bounder;
  std::unique_ptr<RegionEnumerator<T>> enumerator;
};

/// Device memory a build may spend. 0 means "ask the device what is free".
struct BuildBudget
{
  std::size_t bytes = 0;
  std::size_t samples_per_region = 1;  ///< leaves with the sampler (§5.2)
};

/**
 * @brief Which of a Composition's roles a build should actually produce.
 *
 * The driver reaches a Composition by the fathom path or by the subdividing
 * path, never both in one iteration, and a role it did not ask for is device
 * memory spent on something nothing will launch. So the request is explicit
 * rather than implied by the composition: an unrequested role comes back
 * null, and the caller's cache decides when to ask again.
 *
 * `sampler` disappears from this once §5.2 makes the sampler a function of
 * the problem rather than of a composition.
 */
struct RoleRequest
{
  bool sampler = false;
  bool bounder = false;
  bool enumerator = false;

  /// What the subdividing path needs: an incumbent and the bounds to prune by.
  static RoleRequest subdividing() { return {true, true, false}; }

  /// What the fathom path needs, and nothing else.
  static RoleRequest enumerating() { return {false, false, true}; }

  static RoleRequest all() { return {true, true, true}; }
};

/**
 * @brief Builds the roles for a problem, and prices them before they exist.
 *
 * `cost_model` is what makes the resolver possible: it answers "what would
 * this cost" from a Problem alone, with no device and no instance, so the
 * shape fit can run in a host-only target (§5.5).
 *
 * Virtual dispatch is fine here and is not worth engineering away: a bundle
 * is built once per distinct composition (a few dozen per solve) and each
 * virtual call on a role wraps a kernel launch plus a device synchronise.
 */
template<typename T>
class RegionBackendFactory
{
public:
  virtual ~RegionBackendFactory() = default;

  virtual BackendCapabilities capabilities() const = 0;
  virtual RegionCostModel cost_model(const dag::Problem<T>& problem) const = 0;

  /// Once per distinct Composition, for the roles `roles` asks for. Roles it
  /// does not ask for come back null; so does `enumerator` for a composition
  /// that is not fully enumerable, or from a backend that cannot enumerate.
  virtual SubdivisionBundle<T> build_subdivision(
      const dag::Problem<T>& problem,
      const Composition& composition,
      const FanOutSpec& fan_out,
      const BuildBudget& budget,
      RoleRequest roles) const = 0;
};

/**
 * @brief A build that did not fit, as the facts the backend actually knows.
 *
 * The backend does not own the advice it prints when it runs out of memory
 * (§5.4, §5.6): composing "which cap would fit instead" means running the
 * resolver's own scan, and a backend calling the resolver is the wrong
 * direction of dependency -- it is also why that advice was wrong twice
 * (RUNTIME_SHAPE.md §6.3, §6.4). So the backend throws these facts and
 * `config::explain_over_budget` writes the report.
 *
 * The three string members are the one concession to that split: the report
 * names the backend's own internals ("DAG-node buffers"), and only the
 * backend can spell those. They are nouns, not advice -- config still owns
 * every sentence they appear in.
 */
struct OverBudget
{
  std::string role;  ///< e.g. "interval graph"; subject of the first sentence
  std::size_t needed_bytes = 0;
  std::size_t budget_bytes = 0;
  std::size_t n_regions = 0;
  std::size_t elements_per_region = 1;  ///< > 1 only when sampling
  std::size_t bytes_per_element = 0;
  std::string element_breakdown;  ///< what those bytes buy, itemised
  std::string element_inventory;  ///< what the per-element cost scales with
  RegionCostModel cost;
  Composition composition;
  FanOutSpec fan_out;
  std::size_t solve_samples_per_region = 1;  ///< the solve-wide setting
  std::size_t sampler_bytes = 0;  ///< already spent; 0 until §5.2
};

/// Thrown by a build that cannot fit its budget. Derives from
/// ResourceExhausted so every existing catch site keeps working; the payload
/// is what lets a catch site that has a ProblemProfile render the full
/// report instead of the one-line summary in what().
class OverBudgetError : public cuminlp::ResourceExhausted
{
public:
  explicit OverBudgetError(OverBudget facts)
      : cuminlp::ResourceExhausted(summarise(facts))
      , facts_(std::make_shared<const OverBudget>(std::move(facts)))
  {
  }

  const OverBudget& facts() const { return *facts_; }

private:
  // The facts, and only the facts: the report's first sentence. A catch site
  // holding a ProblemProfile replaces this with the full explanation.
  static std::string summarise(const OverBudget& facts)
  {
    return facts.role + " needs " + detail::format_bytes(facts.needed_bytes)
        + " of device memory, but only "
        + detail::format_bytes(facts.budget_bytes) + " is available.";
  }

  // shared_ptr, not a value member: an exception must stay copyable through
  // std::rethrow_exception and OverBudget is not a trivial payload.
  std::shared_ptr<const OverBudget> facts_;
};

}  // namespace cuminlp::backend
