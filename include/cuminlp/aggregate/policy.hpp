#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/aggregate/partition.hpp"
#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/bisection_budget.hpp"
#include "cuminlp/region/composition.hpp"

// The aggregate search's partitioning policy (design/AGGREGATE_BOUNDING.md
// §5.2): which `k` children a node has, and how each is subdivided for
// bounding. Host-only; no CUDA type is in reach.
namespace cuminlp::aggregate
{

/**
 * @brief Splits a node into `k` children and each child into `M` subregions.
 *
 * `branch` carries a purity contract and `refine` does not, and that
 * asymmetry is the whole reason they are separate calls (§5.1):
 * `search::Node`-style `sidx` decoding re-invokes `branch` on a
 * reconstructed parent box and must get the same answer, whereas nothing is
 * ever decoded against `refine`, so it is free to depend on depth, device
 * state, or the incumbent.
 */
template<typename T>
class AggregatePolicy
{
public:
  virtual ~AggregatePolicy() = default;

  /// `k` regions. **Must** be a pure function of `(box, var_kinds)` and
  /// whatever frozen calibration the policy was built with -- a node's child
  /// box is re-derived by calling this again on its parent.
  virtual region::SlotAssignment branch(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const = 0;

  /// The full partition, `branch` included. `depth` is offered because
  /// nothing decodes against `refine`, so a policy may legitimately spend its
  /// budget differently deeper in the tree; the default policy below ignores
  /// it, since `N` is fixed solve-wide (§5.2, §14.2).
  virtual AggregatePartition partition(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds,
      std::size_t depth) const = 0;

  virtual std::size_t branch_fan_out() const = 0;
  virtual std::size_t refine_fan_out() const = 0;
};

/**
 * @brief The default policy: branch `k` ways on one variable, refine the
 *        rest under a bisection budget.
 *
 * ### Why the branch takes exactly one variable
 *
 * The design doc's §5.3 originally had the branch distribute `log2(k)`
 * bisections across variables the way `BisectionBudget` does. That does not
 * survive contact with §3.2's layout requirement. Contiguous child segments
 * need every branch digit to be more significant than every refine digit,
 * which holds when the two halves act on disjoint variables -- but a box with
 * few live variables can have the branch consume all of them, leaving the
 * refine nothing and the launch short of `N` regions. Recovering that by
 * letting the two share a variable (nesting) is only contiguity-safe for
 * *one* shared variable, since a shared variable's branch digit and refine
 * digit are adjacent in significance and other slots' digits fall between
 * them.
 *
 * Branching on exactly one variable makes both problems vanish: the branch
 * uses one slot, so at most one variable is ever shared, and it is always the
 * most significant. It is also the classical B&B formulation, and it costs
 * very little here -- the branch decides tree *shape*, while bound quality
 * comes almost entirely from the refine, which still spends a budget of
 * `log2(N/k)` across every other variable.
 *
 * ### Why the branch slot always partitions, never enumerates
 *
 * `decode::slot_bounds` splits `Continuous`/`IntegerPartition` uniformly and
 * maps `IntegerEnumerate`/`BinaryEnumerate` to single lattice points. Only a
 * uniform split nests: the `c`-th of `k` equal parts of a variable is exactly
 * parts `[c*M, (c+1)*M)` of the same variable split `k*M` ways, so the host's
 * `k`-way branch decode and the device's `k*M`-way launch decode agree
 * bitwise. An enumerating branch slot has no such property, and a `k`-way
 * enumerate over a domain smaller than `k` would also break `Π widths == k`.
 *
 * **Known limitation, to be measured (§14.4):** on a box whose only live
 * variables are Binary, the branch splits a `[0, 1]` domain `k` ways, and for
 * `k = 4` two of the four children contain no integer point. Sound -- an
 * empty child's relaxation is a bound over a region holding no feasible
 * point, which can only be weak, never wrong -- but wasteful, and binary-heavy
 * instances are common in MINLPLib. The fix (letting the branch enumerate
 * several binaries when nesting is not needed) is deliberately not built
 * speculatively; stage 6 says whether it is worth it.
 */
template<typename T>
class WidestBranchPolicy : public AggregatePolicy<T>
{
public:
  /**
   * @param branch_fan_out `k`, at least 2 and a power of two so that `M =
   *        N / k` is exact.
   * @param total_fan_out `N`, a power of two, strictly greater than `k`.
   * @param relative_width forwarded to the refine's `BisectionBudget`, where
   *        it compares each candidate's live width against its own width at
   *        the start of that call rather than the raw absolute width
   *        (`--policy=relative-bisection-budget`'s metric). It does **not**
   *        reach the branch's choice of variable: "relative" there would have
   *        to mean relative to the root domain, and the root box is not in
   *        this policy's scope. The branch picks by absolute live width.
   */
  WidestBranchPolicy(std::size_t branch_fan_out,
                     std::size_t total_fan_out,
                     config::SearchCalibration calibration = {},
                     bool relative_width = false)
      : k_(branch_fan_out)
      , n_(total_fan_out)
      , relative_width_(relative_width)
  {
    if (k_ < 2 || !is_power_of_two(k_)) {
      throw InvalidConfiguration(
          "branch fan-out k must be a power of two and at least 2");
    }
    if (n_ <= k_ || !is_power_of_two(n_)) {
      throw InvalidConfiguration(
          "total fan-out N must be a power of two strictly greater than k, so "
          "that M = N / k is an exact power of two and the refine has a "
          "budget of at least one bisection");
    }
    m_ = n_ / k_;
    refine_ =
        std::make_shared<const policy::BisectionBudgetCompositionPolicy<T>>(
            log2_exact(m_), calibration, relative_width);
  }

  std::size_t branch_fan_out() const override { return k_; }

  std::size_t refine_fan_out() const override { return m_; }

  std::size_t total_fan_out() const { return n_; }

  /// See the constructor: this describes the refine's metric only.
  bool relative_width() const { return relative_width_; }

  region::SlotAssignment branch(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const override
  {
    if (box.size() != var_kinds.size()) {
      throw ShapeMismatch(
          "AggregatePolicy::branch: box and var_kinds differ in length");
    }
    std::optional<std::size_t> const vid = widest_live(box);
    if (!vid) {
      return region::SlotAssignment {};  // fully resolved: a leaf
    }

    region::SlotAssignment out;
    out.composition.kinds.push_back(partition_kind(var_kinds[*vid]));
    out.var_ids.push_back(*vid);
    out.widths.push_back(k_);
    // A uniform split of a live domain never produces a duplicate child, so
    // domain_sizes mirrors widths and region::is_duplicate_child is false for
    // every branch child.
    out.domain_sizes.push_back(k_);
    return out;
  }

  AggregatePartition partition(std::span<const cu::interval<T>> box,
                               std::span<const model::VarKind> var_kinds,
                               std::size_t depth) const override
  {
    (void)depth;  // N is fixed solve-wide; see the class comment.

    AggregatePartition out;
    out.branch = branch(box, var_kinds);
    out.branch_fan_out = k_;
    out.refine_fan_out = m_;
    if (out.branch.size() == 0) {
      return out;  // leaf; launch stays empty
    }

    std::size_t const branch_vid = out.branch.var_ids[0];
    if (live_count(box) == 1) {
      // Nested: the branch took the only live variable, so the refine has
      // nothing disjoint to act on. One slot of width N on that variable,
      // under the branch's own (partitioning) kind -- see AggregatePartition's
      // comment for why that keeps child_id = r / M exact.
      out.nested = true;
      out.launch.composition.kinds.push_back(out.branch.composition[0]);
      out.launch.var_ids.push_back(branch_vid);
      out.launch.widths.push_back(n_);
      out.launch.domain_sizes.push_back(n_);
      return out;
    }

    // Collapsing the branch's variable to a point is how it is withheld from
    // the refine: BisectionBudgetCompositionPolicy skips any dimension that
    // is not live, so this needs no change to that policy and reuses its
    // greedy allocation exactly.
    std::vector<cu::interval<T>> masked(box.begin(), box.end());
    masked[branch_vid].ub = masked[branch_vid].lb;

    out.refine = refine_->choose(masked, var_kinds);
    pad_refine(box, var_kinds, branch_vid, out.refine);
    out.launch = concatenate(out.refine, out.branch);
    return out;
  }

private:
  static bool is_power_of_two(std::size_t v)
  {
    return v != 0 && (v & (v - 1)) == 0;
  }

  static std::size_t log2_exact(std::size_t v)
  {
    std::size_t bits = 0;
    while (v > 1) {
      v >>= 1;
      ++bits;
    }
    return bits;
  }

  /// Continuous stays Continuous; every discrete kind partitions rather than
  /// enumerates, for the nesting reason in the class comment. There is no
  /// BinaryPartition kind, so a Binary uses IntegerPartition -- its bounds
  /// are integer-valued either way.
  static region::SlotKind partition_kind(model::VarKind kind)
  {
    return kind == model::VarKind::Continuous
        ? region::SlotKind::Continuous
        : region::SlotKind::IntegerPartition;
  }

  /**
   * @brief Give every remaining live variable a width-1 slot.
   *
   * Not cosmetic, and not free: it exists because the *graph* is keyed on the
   * launch composition, and `BisectionBudget` emits a slot only for a variable
   * it actually gave a bisection to. How many variables that is depends on the
   * box, so the composition's length moves from node to node -- on ex8_6_2 it
   * oscillated between 9 and 12 slots, rebuilding the bounder graph on 120 of
   * 122 builds and reallocating 23 GB each time. Padding pins the length to
   * the live-variable count, which is far more stable, and the rebuild count
   * fell to single digits.
   *
   * A width-1 `Continuous`/`IntegerPartition` slot is an exact identity:
   * `slot_bounds` computes `width = (ub - lb) / 1` and, since
   * `part + 1 == fan_out`, snaps the upper bound to `parent.ub` verbatim -- so
   * the decoded bound is the parent's, bit for bit. `Π widths` is unchanged,
   * and the padded variables are live and distinct from both the branch
   * variable and each other, so CompositionPolicy's contract still holds.
   *
   * The cost is a wider `apply_slots_kernel` grid (slot_count x N rather than
   * a shorter one) for a kernel that writes one interval per thread. That is
   * cheap against a 23 GB reallocation.
   */
  void pad_refine(std::span<const cu::interval<T>> box,
                  std::span<const model::VarKind> var_kinds,
                  std::size_t branch_vid,
                  region::SlotAssignment& refine) const
  {
    // One slot for the branch, so the refine may fill at most kMaxSlots - 1.
    std::size_t const cap = config::kMaxSlots - 1;
    for (std::size_t vid = 0; vid < box.size() && refine.size() < cap; ++vid) {
      if (vid == branch_vid || !(box[vid].ub > box[vid].lb)) {
        continue;
      }
      if (std::find(refine.var_ids.begin(), refine.var_ids.end(), vid)
          != refine.var_ids.end())
      {
        continue;
      }
      refine.composition.kinds.push_back(partition_kind(var_kinds[vid]));
      refine.var_ids.push_back(vid);
      refine.widths.push_back(1);
      refine.domain_sizes.push_back(1);
    }
  }

  static std::size_t live_count(std::span<const cu::interval<T>> box)
  {
    std::size_t live = 0;
    for (const auto& b : box) {
      if (b.ub > b.lb) {
        ++live;
      }
    }
    return live;
  }

  /// The widest live dimension by absolute width. Ties go to the lowest
  /// index -- deterministic, and hence pure, which `branch` requires.
  static std::optional<std::size_t> widest_live(
      std::span<const cu::interval<T>> box)
  {
    std::optional<std::size_t> best;
    T best_key = T {};
    for (std::size_t i = 0; i < box.size(); ++i) {
      if (!(box[i].ub > box[i].lb)) {
        continue;
      }
      T const key = box[i].ub - box[i].lb;
      if (!best || key > best_key) {
        best = i;
        best_key = key;
      }
    }
    return best;
  }

  std::size_t k_;
  std::size_t n_;
  std::size_t m_ = 1;
  bool relative_width_;
  std::shared_ptr<const policy::BisectionBudgetCompositionPolicy<T>> refine_;
};

}  // namespace cuminlp::aggregate
