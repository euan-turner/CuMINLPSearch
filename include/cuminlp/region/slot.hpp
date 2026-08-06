#pragma once

#include <cstdint>

// Which operation a slot in a Composition performs on its assigned variable
// this iteration -- see region/composition.hpp.
namespace cuminlp::region
{

// This is what fixes a replayed CUDA graph's fan-out/shape, so each distinct
// sequence of SlotKinds corresponds to its own pre-built graph.
enum class SlotKind : std::uint8_t
{
  Continuous,
  IntegerPartition,
  IntegerEnumerate,
  BinaryEnumerate,
};

inline const char* slot_kind_name(SlotKind kind)
{
  switch (kind) {
    case SlotKind::Continuous:
      return "Continuous";
    case SlotKind::IntegerPartition:
      return "IntegerPartition";
    case SlotKind::IntegerEnumerate:
      return "IntegerEnumerate";
    case SlotKind::BinaryEnumerate:
      return "BinaryEnumerate";
  }
  return "?";
}

}  // namespace cuminlp::region
