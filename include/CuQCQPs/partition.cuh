#pragma once

#include "search.hpp"

namespace cuqcqps::partition
{

// The bounding kernel receives a single parent interval, which is stored in
// constant memory as materialising a subinterval per thread does not scale to
// the 10^6 variables required by QPLib The bounds for each variable are
// computed on-demand by each thread, which is O(nnz(Q)).
// TODO: revisit this for smaller problems

// TODO: use CycleSize template more effectively.
/**
 * @brief Per-thread cycling context. Computed per thread from its global index
 *        and used to derive each dimension's sub-interval bounds on-demand.
 *        Needs to be register-resident for efficinecy.
 *
 * @tparam CycleSize Number of dimensions being partitioned
 */
template<std::size_t CycleSize>
struct CycleContext
{
  int part[CycleSize];  // which partition along each partitioned dimension
  int cycleStart;
  int cycleSize;
  int partitionNum;
};

/**
 * @brief Create the per-thread cycling context for a thread
 *
 * @tparam CycleSize        Number of dimensions being partitioned
 * @tparam PartitionNum     Number of partitions per dimension
 * @param tid               Global thread index
 * @param cycleStart        First dimension being partitioned
 * @return CycleContext
 */
template<std::size_t CycleSize, std::size_t PartitionNum>
__device__ CycleContext<CycleSize> make_cycle_context(std::size_t tid,
                                                      int cycleStart)
{
  CycleContext<CycleSize> ctx {};
  ctx.cycleStart = cycleStart;
  ctx.cycleSize = CycleSize;
  ctx.partitionNum = PartitionNum;

  std::size_t idx = tid;
  for (int j = 0; j < CycleSize; ++j) {
    ctx.part[j] = idx % PartitionNum;
    idx /= PartitionNum;
  }
  return ctx;
}

/**
 * @brief Calculate the thread's bounds for a specified dimension.
 *
 * @tparam T Precision used (float/double)
 * @tparam CycleSize Number of dimensions to partition
 * @param ctx CycleContext for the thread
 * @param interval Parent interval
 * @param dim Dimension to bound
 * @param bound Result parameter: bounds for dim
 */
template<typename T, std::size_t CycleSize>
__device__ void get_bounds(const CycleContext<CycleSize>& ctx,
                           const cu::interval<T>* interval,
                           std::size_t dim,
                           cu::interval<T>& bound)
{
  if (dim >= ctx.cycleStart && dim < ctx.cycleStart + ctx.cycleSize) {
    std::size_t i = dim - ctx.cycleStart;
    T width = (interval[dim].ub - interval[dim].lb) / ctx.partitionNum;

    bound.lb = interval[dim].lb + width * ctx.part[i];
    bound.ub = interval[dim].lb + width * (ctx.part[i] + 1);
  } else {
    bound.lb = interval[dim].lb;
    bound.ub = interval[dim].ub;
  }
}

}  // namespace cuqcqps::partition
