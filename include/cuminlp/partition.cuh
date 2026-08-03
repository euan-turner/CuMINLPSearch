#pragma once

#include <cstdint>

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
 * `Capacity` is a compile-time bound; `count` is how many slots are actually
 * live, so one instantiated capacity serves every narrower assignment (see
 * design/RUNTIME_SHAPE.md). Only `count` entries of each array are ever
 * read.
 *
 * Field widths are chosen for the register file, not for convenience: this
 * whole struct is per-thread and register-resident, so every byte is
 * multiplied by the thread count. `std::size_t var_ids` and `int` part/
 * fan_out cost 20 bytes per slot (5 registers), which put capacity 64 at 320
 * registers -- past the 255-per-thread hard cap, i.e. unreachable. At 13
 * bytes per slot (3.25 registers) capacity 64 costs 208 and fits.
 *
 * fan_out stays 32-bit rather than 16: a single slot legitimately exceeds
 * 65535 children (source/power_series.cu bisects one variable 10000 ways,
 * and --partition-num is unbounded above), so a 16-bit radix would silently
 * wrap on a configuration the solver is otherwise happy to run.
 *
 * @tparam Capacity Maximum number of dimensions cycled in one iteration
 */
template<std::size_t Capacity>
struct SlotContext
{
  std::uint32_t part[Capacity];  // this thread's index into each slot's fan-out
  std::uint32_t var_ids[Capacity];  // which variable each slot acts on
  std::uint32_t fan_out[Capacity];  // radix used to decode `part`
  SlotKind kind[Capacity];  // operation each slot performs (uint8_t-backed)
  std::uint32_t count;  // slots in use; entries past this are unread
};

/**
 * @brief Create the per-thread slot context for a thread
 *
 * @tparam Capacity     Compile-time slot capacity
 * @param tid           Global thread index
 * @param slot_var_ids  Which variable each live slot acts on
 * @param slot_fan_out  Fan-out (radix) of each slot, from Composition +
 * FanOutSpec
 * @param slot_kind     Operation each slot performs
 * @param slot_count    Live slots; the padding tail past it is skipped
 * @return SlotContext
 */
template<std::size_t Capacity>
__device__ SlotContext<Capacity> make_slot_context(
    std::size_t tid,
    const std::size_t* __restrict__ slot_var_ids,
    const std::uint32_t* __restrict__ slot_fan_out,
    const SlotKind* __restrict__ slot_kind,
    std::size_t slot_count)
{
  SlotContext<Capacity> ctx {};
  ctx.count = static_cast<std::uint32_t>(slot_count);

  std::size_t idx = tid;
  // Bounded by Capacity so the trip count is still statically known (the
  // loop unrolls), but exits at `count` so a narrow assignment on a wide
  // rung doesn't walk its padding tail.
#pragma unroll
  for (std::size_t j = 0; j < Capacity; ++j) {
    if (j >= ctx.count) {
      break;
    }
    ctx.var_ids[j] = static_cast<std::uint32_t>(slot_var_ids[j]);
    ctx.fan_out[j] = slot_fan_out[j];
    ctx.kind[j] = slot_kind[j];
    ctx.part[j] =
        static_cast<std::uint32_t>(idx % static_cast<std::size_t>(ctx.fan_out[j]));
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
 * @tparam Capacity Compile-time slot capacity
 * @param ctx SlotContext for the thread
 * @param interval Parent interval
 * @param dim Dimension to bound
 * @param bound Result parameter: bounds for dim
 */
template<typename T, std::size_t Capacity>
__device__ void get_slot_bounds(const SlotContext<Capacity>& ctx,
                                const cu::interval<T>* interval,
                                std::size_t dim,
                                cu::interval<T>& bound)
{
  // Only live slots: the padding tail never matches a real dim anyway (its
  // var_ids entries are left zeroed), so stopping at count is a pure saving.
#pragma unroll
  for (std::size_t j = 0; j < Capacity; ++j) {
    if (j >= ctx.count) {
      break;
    }
    if (ctx.var_ids[j] != dim) {
      continue;
    }

    // Delegates to the same decode CompositionInterval::materialise uses
    // host-side (search.hpp), so host/device agreement is structural rather
    // than two hand-written implementations that happen to match (see
    // TEST_EXTENSION.md and slot_decode.hpp).
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
