#pragma once

#include <cstdint>
#include <limits>

namespace cuminlp
{

// Shared bookkeeping (global bounds, tolerance, iteration
// budget) for every concrete driver. Concrete drivers -- FixedRosenbrockDriver
// (fixed_rosenbrock_driver.hpp), GraphDriver (graph_driver.cuh), and later
// JITKernelDriver -- derive from this and each define their own solve(),
// since how a problem is supplied (hardcoded vs. a Problem instance) differs
// per driver and isn't expressible as one shared virtual signature.
class driver
{
public:
  explicit driver(uint32_t iter_limit = 1000000, double tolerance = 1e-9)
      : GUB_(std::numeric_limits<double>::max())
      , GLB_(std::numeric_limits<double>::lowest())
      , tolerance_(tolerance)
      , iter_limit_(iter_limit)
      , iter_idx_(0)
  {
  }

  virtual ~driver() = default;

protected:
  double GUB_;
  double GLB_;
  double tolerance_;
  uint32_t iter_limit_;
  uint32_t iter_idx_;
};

}  // namespace cuminlp
