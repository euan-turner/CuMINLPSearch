#include "cuminlp/cuminlp.hpp"
#include "cuminlp/dag.hpp"

auto main() -> int
{
  cuminlp::driver drv;
  drv.solve();

  return 0;
}
