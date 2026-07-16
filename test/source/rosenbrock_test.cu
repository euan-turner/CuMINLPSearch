#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "CuQCQPs/interval.hpp"
#include "CuQCQPs/rosenbrock.cuh"
#include "cpu_rounding.hpp"

using namespace interval;

// GPU arithmetic can contract multiply-adds differently than the host
// compiler, so results are compared with a relative tolerance rather than
// bit-for-bit. float accumulates more rounding noise than double over the
// DIMS-1 term sum, hence the looser tolerance.
template<typename T>
constexpr T rosenbrock_tolerance()
{
  return std::is_same_v<T, float> ? static_cast<T>(1e-3) : static_cast<T>(1e-9);
}

// Independent CPU implementation of interval_rosenbrock_kernel (rosenbrock.cuh),
// using CpuRounding instead of the kernel's CudaRounding.
template<typename T>
Bounds<T> interval_rosenbrock_cpu(const Interval<T>& iv)
{
  std::size_t const DIMS = iv.bounds.size();
  Bounds<T> res(0);
  for (std::size_t i = 0; i < DIMS - 1; ++i) {
    Bounds<T> xi = iv.bounds[i];
    Bounds<T> xi1 = iv.bounds[i + 1];

    Bounds<T> a = sub_bounds<T, CpuRounding>(sqr_bound<T, CpuRounding>(xi), xi1);
    Bounds<T> b = scal_sub_bound<T, CpuRounding>(xi, static_cast<T>(1));

    Bounds<T> c = scal_mul_bound<T, CpuRounding>(sqr_bound<T, CpuRounding>(a), static_cast<T>(100));
    Bounds<T> d = sqr_bound<T, CpuRounding>(b);
    Bounds<T> e = add_bounds<T, CpuRounding>(c, d);
    res = add_bounds<T, CpuRounding>(res, e);
  }
  return res;
}

TEMPLATE_TEST_CASE(
    "rosenbrock_kernel matches the CPU reference across a batch of points",
    "[rosenbrock]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 50;
  constexpr std::size_t N = 50;

  std::mt19937 rng(12345);
  std::uniform_real_distribution<T> dist(static_cast<T>(-30), static_cast<T>(30));

  std::vector<Point<T>> points(N);
  for (auto& p : points) {
    p.elems.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      p.elems[d] = dist(rng);
    }
  }

  auto const gpu_res = launch_rosenbrock<T>(points);
  REQUIRE(gpu_res.size() == N);

  auto const tol = rosenbrock_tolerance<T>();
  for (std::size_t i = 0; i < N; ++i) {
    T const expected = rosenbrock<T>(points[i]);
    CAPTURE(i);
    REQUIRE(gpu_res[i] == Catch::Approx(expected).epsilon(tol));
  }
}

TEMPLATE_TEST_CASE(
    "interval_rosenbrock_kernel matches the CPU rounding oracle and encloses "
    "point evaluations",
    "[rosenbrock][interval]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 50;
  constexpr std::size_t N = 50;

  std::mt19937 rng(54321);
  std::uniform_real_distribution<T> dist(static_cast<T>(-30), static_cast<T>(30));

  std::vector<Interval<T>> intervals(N);
  for (auto& iv : intervals) {
    iv.bounds.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv.bounds[d] = Bounds<T>(std::min(a, b), std::max(a, b));
    }
  }

  auto const gpu_res = launch_interval_rosenbrock<T>(intervals);
  REQUIRE(gpu_res.size() == N);

  SECTION("matches the CPU rounding oracle bit-for-bit")
  {
    for (std::size_t i = 0; i < N; ++i) {
      auto const cpu_res = interval_rosenbrock_cpu<T>(intervals[i]);
      CAPTURE(i);
      CHECK(gpu_res[i].lower == cpu_res.lower);
      CHECK(gpu_res[i].upper == cpu_res.upper);
    }
  }

  SECTION("encloses sampled point evaluations")
  {
    std::uniform_real_distribution<T> unit(static_cast<T>(0), static_cast<T>(1));
    constexpr int SAMPLES_PER_INTERVAL = 20;
    for (std::size_t i = 0; i < N; ++i) {
      for (int s = 0; s < SAMPLES_PER_INTERVAL; ++s) {
        Point<T> p;
        p.elems.resize(DIMS);
        for (std::size_t d = 0; d < DIMS; ++d) {
          Bounds<T> const& b = intervals[i].bounds[d];
          T const t = unit(rng);
          p.elems[d] = b.lower + t * (b.upper - b.lower);
        }
        T const value = rosenbrock<T>(p);
        CAPTURE(i, s);
        CHECK(value >= gpu_res[i].lower);
        CHECK(value <= gpu_res[i].upper);
      }
    }
  }
}
