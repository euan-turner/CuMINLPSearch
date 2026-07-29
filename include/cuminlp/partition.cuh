#pragma once

#include "composition_policy.hpp"
#include "search.hpp"
#include "slot_decode.hpp"

namespace cuminlp::partition
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

/**
 * @brief Per-thread slot context for a Composition/SlotAssignment (see
 *        composition_policy.hpp): unlike CycleContext above, slots act on an
 *        explicit, possibly non-contiguous set of variables, each with its
 *        own fan-out and operation (bisect vs enumerate), rather than a
 *        uniform PartitionNum over a contiguous cycle_start block. Used by
 *        GraphReplay (graph_replay.cuh); CycleContext/get_bounds above remain
 *        the hand-rolled FixedRosenbrockDriver/rosenbrock.cuh's own, simpler
 *        model and are untouched by this.
 *
 * @tparam CycleSize Number of dimensions being cycled this iteration
 */
template<std::size_t CycleSize>
struct SlotContext
{
  int part[CycleSize];  // this thread's index into each slot's fan-out
  std::size_t var_ids[CycleSize];  // which variable each slot acts on
  int fan_out[CycleSize];  // radix used to decode `part` for each slot
  SlotKind kind[CycleSize];  // operation each slot performs
};

/**
 * @brief Create the per-thread slot context for a thread
 *
 * @tparam CycleSize    Number of dimensions being cycled
 * @param tid           Global thread index
 * @param slot_var_ids  Which variable each of the CycleSize slots acts on
 * @param slot_fan_out  Fan-out (radix) of each slot, from Composition +
 * PartitionNum
 * @param slot_kind     Operation each slot performs
 * @return SlotContext
 */
template<std::size_t CycleSize>
__device__ SlotContext<CycleSize> make_slot_context(
    std::size_t tid,
    const std::size_t* __restrict__ slot_var_ids,
    const int* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind)
{
  SlotContext<CycleSize> ctx {};

  std::size_t idx = tid;
  for (std::size_t j = 0; j < CycleSize; ++j) {
    ctx.var_ids[j] = slot_var_ids[j];
    ctx.fan_out[j] = slot_fan_out[j];
    ctx.kind[j] = slot_kind[j];
    ctx.part[j] =
        static_cast<int>(idx % static_cast<std::size_t>(ctx.fan_out[j]));
    idx /= static_cast<std::size_t>(ctx.fan_out[j]);
  }
  return ctx;
}

/**
 * @brief Calculate the thread's bounds for a specified dimension, dispatching
 *        per-slot on SlotKind (bisect for Continuous/IntegerBisect, point
 *        value for IntegerEnumerate/BinaryEnumerate).
 *
 * @tparam T Precision used (float/double)
 * @tparam CycleSize Number of dimensions being cycled
 * @param ctx SlotContext for the thread
 * @param interval Parent interval
 * @param dim Dimension to bound
 * @param bound Result parameter: bounds for dim
 */
template<typename T, std::size_t CycleSize>
__device__ void get_slot_bounds(const SlotContext<CycleSize>& ctx,
                                const cu::interval<T>* interval,
                                std::size_t dim,
                                cu::interval<T>& bound)
{
  for (std::size_t j = 0; j < CycleSize; ++j) {
    if (ctx.var_ids[j] != dim) {
      continue;
    }

    // Delegates to the same decode CompositionInterval::materialise uses
    // host-side (search.hpp), so host/device agreement is structural rather
    // than two hand-written implementations that happen to match (see
    // TEST_EXTENSION.md §4a and slot_decode.hpp).
    cuminlp::decode::slot_bounds<T>(ctx.kind[j],
                                    interval[dim],
                                    static_cast<std::size_t>(ctx.part[j]),
                                    static_cast<std::size_t>(ctx.fan_out[j]),
                                    bound);
    return;
  }

  // Not one of this iteration's cycled dimensions -- unchanged from the
  // parent.
  bound.lb = interval[dim].lb;
  bound.ub = interval[dim].ub;
}

}  // namespace cuminlp::partition
