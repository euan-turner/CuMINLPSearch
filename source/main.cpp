#include "CuQCQPs/CuQCQPs.hpp"

auto main() -> int
{
  cuqcqps::driver drv;
  drv.solve();

  return 0;
}
