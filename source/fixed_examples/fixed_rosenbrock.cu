#include "fixed_rosenbrock_driver.hpp"

auto main() -> int
{
  cuminlp::rosenbrock::FixedRosenbrockDriver drv;
  drv.solve();

  return 0;
}