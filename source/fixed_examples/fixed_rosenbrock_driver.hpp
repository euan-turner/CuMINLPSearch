#pragma once

#include "cuminlp/cuminlp.hpp"

namespace cuminlp::rosenbrock
{

// Branch-and-bound driver for a hardcoded N-dimensional Rosenbrock instance
// over [-30, 30]^N (see rosenbrock.cuh), evaluated with the hand-written
// per-op kernels there rather than a Problem-driven CUDA graph.
class FixedRosenbrockDriver : public driver
{
public:
  using driver::driver;

  auto solve() -> double;
};

}  // namespace cuminlp::rosenbrock