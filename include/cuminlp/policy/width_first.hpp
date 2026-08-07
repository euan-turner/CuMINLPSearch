#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/policy.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"

namespace cuminlp::policy
{

// Width-first, stateless composition policy: fills slots from unresolved
// (i.e. non-degenerate, lb < ub) variables, widest live domain first,
// regardless of kind.
//
// Continuous variables always partition -- there's no other option for
// them. An integer variable only enumerates once its range is the box's
// unique widest live domain; otherwise it partitions just like a continuous
// variable, so it keeps shrinking in step with the rest of the box rather
// than jumping straight to enumeration while something wider is still
// unresolved. A binary variable (always exactly width 1 while live) only
// enumerates once every other live variable is already down to width <= 1,
// or is itself an integer about to collapse to a point this same cycle
// (IntegerEnumerate) -- binaries are therefore the last thing this policy
// ever touches.
template<typename T>
class WidthFirstCompositionPolicy : public CompositionPolicy<T>
{
public:
  explicit WidthFirstCompositionPolicy(
      region::FanOutSpec fan_out, config::SearchCalibration calibration = {})
      : CompositionPolicy<T>(calibration)
      , fan_out_(fan_out)
  {
  }

  using CompositionPolicy<T>::max_cycle_size;

  const region::FanOutSpec& fan_out() const { return fan_out_; }

  std::size_t n_regions(const region::Composition& composition) const override
  {
    return region::composition_fan_out(composition, fan_out_);
  }

  region::SlotAssignment choose(
      std::span<const cu::interval<T>> box,
      std::span<const model::VarKind> var_kinds) const override
  {
    assert(box.size() == var_kinds.size());
    if (var_kinds.empty()) {
      throw ShapeMismatch(
          "CompositionPolicy::choose called with an empty var_kinds span; "
          "there are no variables to assign to any slot");
    }

    std::vector<std::size_t> live;
    for (std::size_t vid = 0; vid < var_kinds.size(); ++vid) {
      if (unresolved(box[vid])) {
        live.push_back(vid);
      }
    }

    // The box's unique widest live domain, if there is one -- an integer
    // only enumerates when it alone holds this width; a tie means some
    // other variable is at least as wide, so it keeps partitioning instead
    // ("less than its range" is a strict comparison). Found via </> only
    // (-Werror=float-equal bans == on T): max_count counts every live var
    // whose width isn't strictly below max_width, i.e. that ties the max.
    T max_width {0};
    for (std::size_t vid : live) {
      T const w = width(box[vid]);
      if (w > max_width) {
        max_width = w;
      }
    }
    std::size_t max_count = 0;
    for (std::size_t vid : live) {
      if (!(width(box[vid]) < max_width)) {
        ++max_count;
      }
    }
    auto is_unique_widest = [&](std::size_t vid)
    { return max_count == 1 && !(width(box[vid]) < max_width); };

    // An integer only enumerates when it's the unique widest live domain
    // *and* enumerating actually covers it: IntegerEnumerate produces
    // exactly enumerate_cap children at ceil(lb)+0..+enumerate_cap-1
    // (region/decode.hpp), so a domain wider than enumerate_cap would
    // silently drop every integer point past that -- unsound, not just
    // slow. A too-wide widest integer partitions instead, same as
    // GreedyEnumCompositionPolicy's own enumerate_cap gate.
    auto integer_would_enumerate = [&](std::size_t vid)
    {
      return is_unique_widest(vid)
          && integer_domain_size(box[vid]) <= fan_out().enumerate_cap();
    };

    // A binary is ready to enumerate once every other live variable is
    // already down to width <= 1, or is an integer that will itself
    // collapse to a point this cycle (IntegerEnumerate) -- both cases leave
    // nothing wider than the binary still standing after this cycle.
    auto binary_ready = [&](std::size_t vid)
    {
      for (std::size_t u : live) {
        if (u == vid) {
          continue;
        }
        bool const collapses_this_cycle = var_kinds[u] == model::VarKind::Integer
            && integer_would_enumerate(u);
        if (width(box[u]) > T {1} && !collapses_this_cycle) {
          return false;
        }
      }
      return true;
    };

    struct Candidate
    {
      std::size_t vid;
      region::SlotKind kind;
      T w;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(live.size());
    for (std::size_t vid : live) {
      switch (var_kinds[vid]) {
        case model::VarKind::Continuous:
          candidates.push_back(
              {vid, region::SlotKind::Continuous, width(box[vid])});
          break;
        case model::VarKind::Integer:
          candidates.push_back(
              {vid,
               integer_would_enumerate(vid) ? region::SlotKind::IntegerEnumerate
                                            : region::SlotKind::IntegerPartition,
               width(box[vid])});
          break;
        case model::VarKind::Binary:
          // No BinaryPartition slot kind exists, so a not-yet-ready binary
          // simply gets no slot this cycle -- it stays live for the next
          // one, same as any other candidate the cap left behind.
          if (binary_ready(vid)) {
            candidates.push_back(
                {vid, region::SlotKind::BinaryEnumerate, width(box[vid])});
          }
          break;
      }
    }

    // Widest first, across every kind alike -- the policy's namesake.
    std::sort(candidates.begin(),
              candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.w > b.w; });

    region::SlotAssignment out {};
    for (const Candidate& c : candidates) {
      if (out.var_ids.size() >= max_cycle_size()) {
        break;
      }
      out.composition.kinds.push_back(c.kind);
      out.var_ids.push_back(c.vid);
      out.widths.push_back(region::slot_fan_out(c.kind, fan_out_));
    }
    return out;
  }

private:
  static bool unresolved(const cu::interval<T>& b) { return b.ub > b.lb; }

  static T width(const cu::interval<T>& b) { return b.ub - b.lb; }

  // Duplicated from GreedyEnumCompositionPolicy rather than shared, to avoid
  // a circular include (greedy_enum.hpp already includes this header for
  // make_policy's WidthFirst case) -- see its own comment for the ceil/floor
  // and empty-domain rationale.
  static std::size_t integer_domain_size(const cu::interval<T>& b)
  {
    T const lo = std::ceil(b.lb);
    T const hi = std::floor(b.ub);
    if (hi < lo) {
      return 0;
    }
    return static_cast<std::size_t>(hi - lo) + 1;
  }

  region::FanOutSpec fan_out_;
};

}  // namespace cuminlp::policy
