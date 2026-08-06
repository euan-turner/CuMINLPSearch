#pragma once

#include <cstddef>

#include "cuminlp/errors.hpp"
#include "cuminlp/region/slot.hpp"

namespace cuminlp::region
{

// How wide each slot fans out. Formerly the PartitionNum/EnumerateCap
// template parameters; they never reached device codegen (the per-slot
// fan-outs they produce are computed host-side and uploaded to a device
// buffer, see GraphReplay::build), so nothing about CUDA graph capture
// required them to be compile-time constants -- only C++ did, and only
// incidentally. As runtime values a single binary can tune them per problem.
//
// The compiler no longer rejects a nonsensical value, so the constructor
// does: partition_num < 2 is the important one, since a fan-out of 1
// subdivides nothing and would let the search loop forever without ever
// narrowing a box.
class FanOutSpec
{
public:
  FanOutSpec(std::size_t partition_num, std::size_t enumerate_cap)
      : partition_num_(partition_num)
      , enumerate_cap_(enumerate_cap)
  {
    if (partition_num < 2) {
      throw InvalidConfiguration(
          "partition_num must be at least 2; a fan-out of 1 subdivides "
          "nothing, so bisection could never narrow a box");
    }
    if (enumerate_cap < 1) {
      throw InvalidConfiguration(
          "enumerate_cap must be at least 1; a slot must produce at least "
          "one child");
    }
  }

  // EnumerateCap used to default to PartitionNum as a template parameter;
  // this keeps that shorthand for callers that don't want the distinction.
  explicit FanOutSpec(std::size_t partition_num)
      : FanOutSpec(partition_num, partition_num)
  {
  }

  std::size_t partition_num() const { return partition_num_; }

  std::size_t enumerate_cap() const { return enumerate_cap_; }

  bool operator==(const FanOutSpec&) const = default;

private:
  std::size_t partition_num_;
  std::size_t enumerate_cap_;
};

// Number of children a slot of the given kind produces. Binary is always
// exactly 2 (its domain is always size 2 until resolved); Continuous/
// IntegerPartition share partition_num (the partition width); IntegerEnumerate
// uses its own enumerate_cap, independent of partition_num, so raising the
// enumerate threshold doesn't force wider partitioning elsewhere.
inline std::size_t slot_fan_out(SlotKind kind, const FanOutSpec& fan_out)
{
  if (kind == SlotKind::BinaryEnumerate) {
    return 2;
  }
  if (kind == SlotKind::IntegerEnumerate) {
    return fan_out.enumerate_cap();
  }
  return fan_out.partition_num();
}

}  // namespace cuminlp::region
