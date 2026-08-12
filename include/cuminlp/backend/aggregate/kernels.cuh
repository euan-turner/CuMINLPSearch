#pragma once

#include <cstddef>

#include <cuinterval/interval.h>

// The aggregate backend's one new kernel (design/AGGREGATE_BOUNDING.md §9).
// Everything else in its graph -- the broadcast/apply root pair, every op
// kernel, every constraint's feasibility_check_kernel, and both objective
// extract kernels -- is backend/graph/kernels.cuh's, unchanged.
namespace cuminlp::backend::aggregate
{

/**
 * @brief Masks both objective bounds by feasibility, in one pass.
 *
 * The counterpart of `graph::mask_infeasible_kernel`, which masks the upper
 * bound only because the existing backend reduces only that. Here both
 * reductions run, over the same mask, so the two masked arrays are written
 * together rather than by two kernels reading `feasible[]` twice.
 *
 * The two sentinels are the identities of the reductions that consume them:
 * `+inf` cannot lower a `Min`, `-inf` cannot raise a `Max`. So an excluded
 * subregion contributes nothing to either aggregate, which is exactly §4.1's
 * masking argument in code.
 *
 * It also makes the child's feasibility fall out of the `Max` for free: if
 * *every* subregion in a child is excluded, that child's `hull_ub` is still
 * `-inf` afterwards. That is what lets §4.3's ambiguity be resolved without a
 * third reduction -- see `AggregateBounderReplay::launch`.
 */
template<typename T>
__global__ void mask_bounds_kernel(const T* __restrict__ obj_lb,
                                   const T* __restrict__ obj_ub,
                                   const unsigned char* __restrict__ feasible,
                                   T* __restrict__ masked_lb,
                                   T* __restrict__ masked_ub,
                                   std::size_t n_elems)
{
  std::size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = gridDim.x * blockDim.x;
  for (std::size_t i = tid; i < n_elems; i += stride) {
    bool const live = feasible[i] != 0;
    masked_lb[i] = live ? obj_lb[i] : T(INFINITY);
    masked_ub[i] = live ? obj_ub[i] : T(-INFINITY);
  }
}

}  // namespace cuminlp::backend::aggregate
