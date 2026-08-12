#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cuminlp/errors.hpp"
#include "cuminlp/region/composition.hpp"

// The branch/refine split (design/AGGREGATE_BOUNDING.md §5) as data: the two
// SlotAssignments a node is partitioned by, and the single concatenated
// assignment the device is actually launched with.
//
// Host-only, and named by `aggregate/bound.hpp` so the backend roles can talk
// about a partition without depending on the policy that produced one.
namespace cuminlp::aggregate
{

/**
 * @brief One node's partition: `k` children, each covered by `M` subregions.
 *
 * The three assignments are redundant by construction and that is the point
 * -- `launch` is what the device decodes, `branch` is what the host decodes a
 * child's `sidx` against, and holding both lets `validate()` state the
 * relationship between them as a checkable invariant rather than a comment.
 *
 * ### Slot order (§3.2)
 *
 * `launch` is `refine`'s slots followed by `branch`'s, in that order. Since
 * `apply_slots_kernel` decodes slot `j`'s digit as
 * `(r / prefix[j]) % widths[j]` with `prefix` the exclusive product, slot 0
 * is *least* significant -- so putting the branch slots last makes the branch
 * digits the most significant ones, and hence makes each child's `M`
 * subregions contiguous in `r`:
 *
 *     r = child_id * M + sub_id,    child_id = r / M
 *
 * which is what lets the epilogue reduce over `k` contiguous segments.
 *
 * ### The nested case
 *
 * When the popped box has exactly one live variable there is nothing left for
 * the refine to act on once the branch has taken it, so the two share that
 * one variable and `launch` holds a single slot of width `k * M` rather than
 * the concatenation of two disjoint slot sets. `branch` still describes a
 * `k`-way split of the same variable under the same kind, so `child_id =
 * r / M` and the host decode both still hold exactly -- which is true only
 * because the branch always uses a *partitioning* kind (uniform split), never
 * an enumerating one. See `WidestBranchPolicy`.
 */
struct AggregatePartition
{
  region::SlotAssignment branch;  ///< k regions; the reproducible half
  region::SlotAssignment refine;  ///< M regions; ephemeral. Empty when nested.
  region::SlotAssignment launch;  ///< N = k * M regions; what the device runs
  std::size_t branch_fan_out = 1;  ///< k
  std::size_t refine_fan_out = 1;  ///< M
  bool nested = false;  ///< branch and refine share their one variable

  std::size_t total_fan_out() const { return branch_fan_out * refine_fan_out; }

  /// A fully-resolved box: no live variable, so nothing to branch on. The
  /// driver treats such a node as a leaf rather than launching for it.
  bool is_leaf() const { return branch.size() == 0; }

  /**
   * @brief Everything §5 and §3.2 claim about a partition, as a check.
   *
   * Cheap enough to call per node in a debug build, and every clause is one
   * the search would otherwise be silently wrong about:
   *
   * - `Π branch.widths == k` and `Π refine.widths == M`, so `Π launch.widths
   *   == N` -- the graph is built for exactly `N` regions and a mismatch
   *   sizes buffers wrongly.
   * - branch and refine `var_ids` are disjoint (unless nested), because
   *   `apply_slots_kernel`'s last-write-wins is correct only when at most one
   *   slot touches each dimension. Two slots on one variable would silently
   *   compute a child's bound over the wrong box.
   * - the branch slots are last in `launch`, without which `child_id = r / M`
   *   is false and every child gets another child's bound.
   *
   * @throws InvalidConfiguration naming the clause that failed.
   */
  void validate() const
  {
    using region::composition_fan_out;

    if (is_leaf()) {
      return;
    }

    if (composition_fan_out(branch.widths) != branch_fan_out) {
      throw InvalidConfiguration(
          "AggregatePartition: product of branch widths does not equal the "
          "branch fan-out k");
    }
    if (composition_fan_out(launch.widths) != total_fan_out()) {
      throw InvalidConfiguration(
          "AggregatePartition: product of launch widths does not equal "
          "N = k * M");
    }

    std::size_t const n_branch = branch.size();
    if (launch.size() < n_branch) {
      throw InvalidConfiguration(
          "AggregatePartition: launch has fewer slots than branch");
    }

    if (nested) {
      if (launch.size() != 1 || branch.size() != 1 || refine.size() != 0) {
        throw InvalidConfiguration(
            "AggregatePartition: a nested partition is exactly one slot "
            "shared by branch and refine");
      }
      if (launch.var_ids[0] != branch.var_ids[0]
          || launch.composition[0] != branch.composition[0])
      {
        throw InvalidConfiguration(
            "AggregatePartition: the nested slot does not match the branch "
            "slot's variable and kind");
      }
      return;
    }

    if (composition_fan_out(refine.widths) != refine_fan_out) {
      throw InvalidConfiguration(
          "AggregatePartition: product of refine widths does not equal the "
          "refine fan-out M");
    }
    if (launch.size() != branch.size() + refine.size()) {
      throw InvalidConfiguration(
          "AggregatePartition: launch is not the concatenation of refine and "
          "branch");
    }

    // §3.2: branch slots occupy the top (most significant) positions.
    std::size_t const offset = launch.size() - n_branch;
    for (std::size_t j = 0; j < n_branch; ++j) {
      if (launch.var_ids[offset + j] != branch.var_ids[j]
          || launch.widths[offset + j] != branch.widths[j]
          || launch.composition[offset + j] != branch.composition[j])
      {
        throw InvalidConfiguration(
            "AggregatePartition: branch slots are not the most significant "
            "slots of launch, so child_id = r / M does not hold");
      }
    }

    for (std::size_t bv : branch.var_ids) {
      if (std::find(refine.var_ids.begin(), refine.var_ids.end(), bv)
          != refine.var_ids.end())
      {
        throw InvalidConfiguration(
            "AggregatePartition: branch and refine act on the same variable "
            "without being nested; apply_slots_kernel would decode one of "
            "them into the wrong box");
      }
    }
  }
};

/// The child a launch-region index belongs to (§3.2). Trivial, and named
/// rather than open-coded so the one place the layout is assumed is greppable
/// from both the host driver and the device epilogue.
inline std::size_t child_of_region(std::size_t r, std::size_t refine_fan_out)
{
  return r / refine_fan_out;
}

/// `refine` slots first, `branch` slots last -- the order §3.2 requires.
/// Kept separate from the policy so a different policy composing the same two
/// halves cannot get the order wrong in a new way.
inline region::SlotAssignment concatenate(const region::SlotAssignment& refine,
                                          const region::SlotAssignment& branch)
{
  region::SlotAssignment out;
  auto append = [&out](const region::SlotAssignment& a)
  {
    for (std::size_t j = 0; j < a.size(); ++j) {
      out.composition.kinds.push_back(a.composition[j]);
      out.var_ids.push_back(a.var_ids[j]);
      out.widths.push_back(a.widths[j]);
      // domain_sizes is parallel to widths or empty; a partition whose halves
      // disagree about that would leave is_duplicate_child reading past the
      // end, so the concatenation always populates it, defaulting to "no
      // rounding" where a half left it empty.
      out.domain_sizes.push_back(a.domain_sizes.empty() ? a.widths[j]
                                                        : a.domain_sizes[j]);
    }
  };
  append(refine);
  append(branch);
  return out;
}

}  // namespace cuminlp::aggregate
