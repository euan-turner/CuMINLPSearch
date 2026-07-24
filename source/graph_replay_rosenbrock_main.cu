// Runs the same Rosenbrock instance as source/main.cpp (cuminlp::driver),
// but through the DAG/CUDA-graph-replay evaluator (cuminlp::rosenbrock::
// graph_driver, include/cuminlp/graph_replay_rosenbrock.cuh) instead of the
// hand-written per-op kernels in include/cuminlp/rosenbrock.cuh. Intended to
// be run side-by-side with cuminlp_exe to compare the two evaluators.

#include "cuminlp/graph_replay_rosenbrock.cuh"

auto main() -> int
{
  cuminlp::rosenbrock::graph_driver drv;
  drv.solve();

  return 0;
}
