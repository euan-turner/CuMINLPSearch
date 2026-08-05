#include <cmath>
#include <limits>
#include <string>

#include "fixed_rosenbrock_driver.hpp"

auto main() -> int
{
  // Smoke test: run a handful of branch-and-bound iterations over the
  // (currently hardcoded) Rosenbrock instance and check the search actually
  // improved on the trivial +inf upper bound it started from. Requires an
  // actual CUDA device at test time.
  cuminlp::rosenbrock::FixedRosenbrockDriver drv(200);
  double const gub = drv.solve();
  if (!std::isfinite(gub) || gub < 0.0
      || gub >= std::numeric_limits<double>::max())
  {
    return 1;
  }

  return 0;
}
