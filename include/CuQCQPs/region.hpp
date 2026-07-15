#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace region
{

// RegionHistory is append only, when a region is selected, the full 2*n
// interval is added may be worth converting to a min-heap in the future

// Pending regions are (iter_idx, subreg_idx, cycl_idx, lb).

// Reconstruction just checks RegionHistory[iter_idx]

struct Region
{
  uint32_t sr_idx;  // index within parent's partition
  uint32_t iter_idx;  // index into region list, corresponds to region's direct
                      // parent
  uint16_t cycl_idx;  // block of dimensions split to produce this region
  double lb;  // objective lower bound in this region
  bool alive;  // lazy deletion flag
};

static_assert(sizeof(Region) == 32);

// TODO: RegionList as a min-heap
struct RegionList
{
  std::vector<Region> regions;

  explicit RegionList(std::size_t initial_size = 0)
      : regions(initial_size)
  {
  }
};

template<std::size_t NDIMS>
struct ExplicitRegion
{
  std::array<double, NDIMS> lower;
  std::array<double, NDIMS> upper;
};

template<std::size_t NDIMS>
struct RegionHistory
{
  std::vector<ExplicitRegion<NDIMS>> regions;

  explicit RegionHistory(std::size_t initial_size = 0)
      : regions(initial_size)
  {
  }
};

template<std::size_t NDIMS>
void materialise_region(const Region& region, ExplicitRegion<NDIMS>& out)
{
  if (region.iter_idx == 0) {
    // first region
    std::fill(std::begin(out.lower),
              std::end(out.lower),
              std::numeric_limits<double>::lowest());
    std::fill(std::begin(out.upper),
              std::end(out.upper),
              std::numeric_limits<double>::max());
  } else {
    // TODO
    throw std::domain_error("TODO");
  }
}

}  // namespace region
