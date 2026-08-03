#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include "cuminlp/errors.hpp"

namespace cuminlp
{

// The slot capacities GraphDriver/GraphReplay are instantiated for.
//
// Capacity is the one search-shape parameter that genuinely has to reach
// device codegen: it is the array bound of partition::SlotContext, which is
// per-thread and register-resident (see design/RUNTIME_SHAPE.md). So
// instead of one value fixed at build time, a small ladder is compiled and
// the nearest rung selected at runtime.
//
// Rounding a request up to the next rung costs nothing but registers,
// because Padding slots have fan-out 1: a capacity-8 context running a
// count-4 assignment produces exactly the children a capacity-4 one would.
// That is what makes capacity and count separable at all.
//
// Nothing below 8 is worth a rung -- other per-thread working state dominates
// at that size. 64 is the top because SlotContext costs ~3.25 registers per
// slot, so 64 slots is ~208 of the 255 registers a thread may hold.
inline constexpr std::array<std::size_t, 4> capacity_ladder = {8, 16, 32, 64};

inline constexpr std::size_t max_capacity()
{
  return capacity_ladder.back();
}

// The rung a solve needing `slots` slots should be built at: the smallest
// one that fits, or 0 if none does.
//
// The zero-returning form exists so a caller that knows its slot count at
// compile time (the hand-written drivers in source/, which pick a fixed
// CYCLE_SIZE) can name its capacity in a constant expression and
// static_assert the result, rather than going through the throwing overload
// below.
inline constexpr std::size_t ladder_rung_or_zero(std::size_t slots)
{
  for (std::size_t rung : capacity_ladder) {
    if (slots <= rung) {
      return rung;
    }
  }
  return 0;
}

// Runtime form: same answer, but a typed failure rather than a sentinel.
inline std::size_t ladder_rung_for(std::size_t slots)
{
  std::size_t const rung = ladder_rung_or_zero(slots);
  if (rung == 0) {
    throw InvalidConfiguration(
        "a slot capacity of " + std::to_string(slots)
        + " is past the widest compiled rung ("
        + std::to_string(max_capacity())
        + "); SlotContext is register-resident, so a wider rung would spill "
          "to local memory rather than simply cost more registers");
  }
  return rung;
}

/**
 * @brief Invoke `f.template operator()<Rung>()` for the smallest ladder rung
 *        that holds `slots`.
 *
 * The bridge between a runtime slot count and the compile-time capacity the
 * kernels need. Callers write one templated lambda and get it instantiated
 * once per rung; which one runs is decided at runtime.
 *
 * Throws InvalidConfiguration if `slots` exceeds the widest rung, rather
 * than silently clamping -- a clamp would run a search with fewer slots than
 * asked for and report nothing about it.
 */
template<typename F>
decltype(auto) dispatch_on_capacity(std::size_t slots, F&& f)
{
  std::size_t const rung = ladder_rung_for(slots);
  if (rung == 8) {
    return std::forward<F>(f).template operator()<8>();
  }
  if (rung == 16) {
    return std::forward<F>(f).template operator()<16>();
  }
  if (rung == 32) {
    return std::forward<F>(f).template operator()<32>();
  }
  return std::forward<F>(f).template operator()<64>();
}

}  // namespace cuminlp
