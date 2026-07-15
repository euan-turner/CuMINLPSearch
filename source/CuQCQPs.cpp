#include <limits>
#include <string>

#include "CuQCQPs/CuQCQPs.hpp"

#include "CuQCQPs/region.hpp"

library_stub::library_stub()
    : m_name {"CuQCQPs"}
{
}

auto library_stub::name() const -> char const*
{
  return m_name.c_str();
}

driver::driver(uint32_t iter_limit)
    : GUB_(std::numeric_limits<double>::max())
    , GLB_(std::numeric_limits<double>::lowest())
    , iter_limit_(iter_limit)
    , iter_idx_(0)
    , cycl_idx_(0)
{
}

auto driver::solve() -> double
{
  // 1. Initialise region list on CPU with a single region
  region::RegionList pending(1000);
  pending.regions[0] = {
      .sr_idx = 0,
      .iter_idx = 0,
      .cycl_idx = 0,
      .lb = std::numeric_limits<double>::lowest(),
      .alive = true,
  };

  // 50 dimensions for now, TODO
  region::RegionHistory<50> region_history(1000);

  // 2. while not finished
  while (iter_idx_ < iter_limit_) {
    // 3. Select and materialise region and send to GPU

    // Find region with least lower bound

    // materialise_region()

    // 4. Kernels
    // 4a. Sample points from within the region

    // 4b. Interval analysis over points

    // 4c. Reduce GUB

    // 4d. Partition into sub-regions

    // 4e. Interval analysis over sub-regions

    // 4f. Store region lower-bound

    // 5. CPU reads regions, updates GUB and prunes

    // 6. Repeat until convergence
    ++iter_idx_;
  }

  return 0.0;
}
