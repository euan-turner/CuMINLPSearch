#pragma once

#include <cstddef>

// Hardware- and user-derived tuning inputs, read once and then frozen for
// the lifetime of a solve.
//
// The freezing is a *correctness* requirement, not a performance choice.
// search::Node::materialise reconstructs a node's box by
// re-invoking policy.choose() on the parent box and decoding sidx against
// the fan-outs that produces. If choose() consulted live device state --
// free memory, occupancy, queue depth -- then the assignment at enqueue time
// and at materialise time could differ, sidx would decode against the wrong
// radix vector, and the search would compute silently wrong boxes: no crash,
// no exception, a plausible-looking wrong optimum. Freezing at solve() entry
// lets the policy adapt to the hardware while staying a pure function of
// (box, var_kinds, calibration). See design/RUNTIME_SHAPE.md.
namespace cuminlp::config
{

struct SearchCalibration
{
  // Upper bound on slots the policy may fill. Never above kMaxSlots.
  std::size_t max_cycle_size = 0;
  std::size_t free_device_bytes = 0;
  std::size_t multiprocessor_count = 0;
};

// Search-shape guard, not a compiled bound: beyond ~64 slots a single
// composition's fan-out product is past anything a device can hold at any
// fan-out, so this is a ceiling on absurdity, in the same sense
// partition_ceiling (config/resolve.hpp) is (design/MODULE_REFACTOR.md §4.7).
inline constexpr std::size_t kMaxSlots = 64;

}  // namespace cuminlp::config
