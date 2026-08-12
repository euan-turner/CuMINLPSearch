#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <cuinterval/interval.h>

#include "cuminlp/aggregate/partition.hpp"
#include "cuminlp/backend/backend.hpp"

// The two roles the aggregate backend plays, named for what they do
// (design/AGGREGATE_BOUNDING.md §9). Deliberately a separate hierarchy from
// backend::RegionBackendFactory rather than a fourth role on it: the two
// backends are compared, not merged, and a shared factory interface would
// force the existing graph backend to answer for a role it does not
// implement (§1).
//
// Host-only by construction, like backend/backend.hpp: this is what
// `aggregate/` is allowed to name, and `aggregate/` must compile without a
// CUDA toolchain.
namespace cuminlp::aggregate
{

/**
 * @brief What one launch learned about one child, reduced on device.
 *
 * The `M` per-subregion verdicts inside a child never reach the host; these
 * three fields are what replaces them (§3.3).
 *
 * `hull_ub` is **not an incumbent** and must never be folded into GUB (§4.5).
 * It is a sound upper bound on the objective over the child's unexcluded
 * part, attained at no known point and not known to be attained at a feasible
 * one. It exists solely to give `SelectionRule::RejectIndex` its width. The
 * field is spelled `hull_ub` rather than `ub` precisely because a struct with
 * `lb` and `ub` side by side invites that mistake once, quietly, and the
 * mistake would corrupt the primal bound.
 */
template<typename T>
struct AggregateBound
{
  /// min over unexcluded subregions of the objective's interval lower bound.
  /// Sound lower bound over the child's feasible part (§4.1). `+inf` when
  /// every subregion was excluded.
  T lb = std::numeric_limits<T>::max();

  /// max over unexcluded subregions of the objective's interval upper bound.
  /// See the warning above.
  T hull_ub = std::numeric_limits<T>::max();

  /// False iff every one of the child's subregions was proven infeasible, in
  /// which case the child holds no feasible point and is discarded outright.
  /// Distinct from `lb == +inf`, which an unbounded relaxation can also
  /// produce (§4.3) -- the backend counts unexcluded subregions to tell them
  /// apart, because only the first is the `infeasible` claim MINLP_STATUS.md
  /// records.
  bool feasible = true;
};

/**
 * @brief One popped node: its box, and the partition chosen for it.
 *
 * `partition` must outlive the call it is passed to; the driver holds it for
 * the iteration.
 */
template<typename T>
struct AggregateRegion
{
  std::span<const cu::interval<T>> box;  ///< n_vars entries
  const AggregatePartition& partition;
};

/**
 * @brief Reduces `N` per-subregion verdicts into `k` child bounds per region.
 *
 * Takes a span of regions rather than one region, from day one, even though
 * the first driver runs with `regions.size() == 1`: batching popped nodes
 * into a single launch is how the per-iteration launch latency stops binding
 * (§14.1), and retrofitting a batch dimension through the graph builder and
 * the epilogue later is substantially more invasive than carrying an unused
 * one now.
 */
template<typename T>
class AggregateBounder
{
public:
  virtual ~AggregateBounder() = default;

  /// `regions.size() * k` entries, region-major: entry `i * k + c` is region
  /// `i`'s child `c`. A view over storage this instance owns, valid until the
  /// next call on it.
  virtual std::span<const AggregateBound<T>> bound_children(
      std::span<const AggregateRegion<T>> regions) = 0;

  /// Subregions evaluated per region per call (`N`). For telemetry: it is the
  /// device work one iteration cost, and it is not otherwise recoverable from
  /// anything that crosses PCIe.
  virtual std::size_t n_subregions() const = 0;
};

/**
 * @brief Draws feasible candidates across a popped node; the source of
 *        incumbents.
 *
 * Reuses `backend::CandidateResult` rather than defining a twin: the shape --
 * found / value / witness / index -- is identical, and two structs that must
 * agree are one more thing to disagree.
 *
 * Unlike the bounder this reduces to a *single* candidate across every region
 * and every child (§6.2). The incumbent is global; which child the best
 * sampled point fell in does not matter, and a per-child reduction would cost
 * `k` times the epilogue for an answer nothing reads.
 */
template<typename T>
class AggregateSampler
{
public:
  virtual ~AggregateSampler() = default;

  /// `salt` is the driver's iteration index, so a revisited box draws fresh
  /// points.
  virtual backend::CandidateResult<T> sample(
      std::span<const AggregateRegion<T>> regions, std::uint64_t salt) = 0;

  virtual std::size_t n_samples() const = 0;
};

}  // namespace cuminlp::aggregate
