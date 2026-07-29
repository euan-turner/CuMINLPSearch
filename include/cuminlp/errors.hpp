#pragma once

#include <stdexcept>
#include <string>

namespace cuminlp {

// Common base so callers can catch broadly; derives from std::runtime_error
// so existing catch(std::runtime_error) sites keep working unmodified.
class error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The expression graph itself is structurally broken (cycle, dangling
// node reference, wrong operand count for an Op, a Const in a position that
// requires a non-degenerate node, etc).
class InvalidDAG : public error {
  using error::error;
};

// The Problem (variables/bounds/kinds/objective/constraints) is malformed,
// independent of whether the underlying graph is well-formed.
class InvalidProblem : public error {
  using error::error;
};

// A caller-supplied argument at a runtime API boundary (post-construction)
// doesn't match an expected size/shape.
class ShapeMismatch : public error {
  using error::error;
};

}  // namespace cuminlp
