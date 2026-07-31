#pragma once

#include <stdexcept>
#include <string>

namespace cuminlp
{

// Common base so callers can catch broadly; derives from std::runtime_error
// so existing catch(std::runtime_error) sites keep working unmodified.
class error : public std::runtime_error
{
  using std::runtime_error::runtime_error;
};

// The expression graph itself is structurally broken (cycle, dangling
// node reference, wrong operand count for an Op, a Const in a position that
// requires a non-degenerate node, etc).
class InvalidDAG : public error
{
  using error::error;
};

// The Problem (variables/bounds/kinds/objective/constraints) is malformed,
// independent of whether the underlying graph is well-formed.
class InvalidProblem : public error
{
  using error::error;
};

// A caller-supplied argument at a runtime API boundary (post-construction)
// doesn't match an expected size/shape.
class ShapeMismatch : public error
{
  using error::error;
};

// A solver tuning value is outside the range that makes sense, independent of
// any Problem: partition_num < 2 (a fan-out of 1 subdivides nothing, so the
// search can never make progress), enumerate_cap < 1, etc. Distinct from
// InvalidProblem because nothing about the model is wrong -- only how the
// caller asked for it to be solved. Introduced when PartitionNum/EnumerateCap
// stopped being template parameters (and so stopped being checkable by the
// compiler) and became runtime configuration.
class InvalidConfiguration : public error
{
  using error::error;
};

// A well-formed request would need more device memory than is available (or
// than the caller budgeted). Its own type rather than InvalidConfiguration
// because the request is legal in principle and may well succeed on a bigger
// GPU, or with a smaller partition_num -- callers can meaningfully retry.
class ResourceExhausted : public error
{
  using error::error;
};

}  // namespace cuminlp
