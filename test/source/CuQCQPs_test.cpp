#include <cmath>
#include <limits>
#include <string>

#include "CuQCQPs/CuQCQPs.hpp"

auto main() -> int
{
  auto const stub = cuqcqps::library_stub {};
  if (std::string("CuQCQPs") != stub.name()) {
    return 1;
  }

  // Smoke test: run a handful of branch-and-bound iterations over the
  // (currently hardcoded) Rosenbrock instance and check the search actually
  // improved on the trivial +inf upper bound it started from. Requires an
  // actual CUDA device at test time.
  cuqcqps::driver drv(200);
  double const gub = drv.solve();
  if (!std::isfinite(gub) || gub < 0.0
      || gub >= std::numeric_limits<double>::max())
  {
    return 1;
  }

  return 0;
}
