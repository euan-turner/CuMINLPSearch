#pragma once

#include <cstddef>
#include <limits>

// Split out of cuda_utils.cuh so host-only code (e.g. config/footprint.hpp,
// config/resolve.hpp) can size a search shape without pulling in
// <cuda_runtime.h> or anything else CUDA-specific. These three functions
// never touched CUDA to begin with; the split is pure code motion.
namespace cuminlp::detail
{

constexpr std::size_t ceil_div(std::size_t x, std::size_t y)
{
  return (x + y - 1) / y;
}

// Saturates at SIZE_MAX instead of wrapping. Used wherever a device
// allocation size is computed from runtime-configurable search parameters:
// a wrapped product would read as comfortably affordable and then size every
// buffer far too small, whereas a saturated one is rejected loudly by the
// memory-budget guard (see GraphReplay::build).
constexpr std::size_t saturating_mul(std::size_t a, std::size_t b)
{
  if (a == 0 || b == 0) {
    return 0;
  }
  if (a > std::numeric_limits<std::size_t>::max() / b) {
    return std::numeric_limits<std::size_t>::max();
  }
  return a * b;
}

// Companion to saturating_mul, for totalling the footprints of several
// already-saturating buffer sizes. Wrapping here would be worse than for a
// product: a sum of three plausible sizes that wraps to a small number reads
// as an easily affordable total.
constexpr std::size_t saturating_add(std::size_t a, std::size_t b)
{
  if (a > std::numeric_limits<std::size_t>::max() - b) {
    return std::numeric_limits<std::size_t>::max();
  }
  return a + b;
}

}  // namespace cuminlp::detail
