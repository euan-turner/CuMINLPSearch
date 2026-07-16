#include <cmath>
#include <cstddef>
#include <limits>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "CuQCQPs/cuinterval.cuh"
#include "CuQCQPs/interval.hpp"
#include "cpu_rounding.hpp"

using namespace cuqcqps::interval;

#define CUDA_CHECK(expr) \
  do { \
    cudaError_t const cuda_check_err_ = (expr); \
    REQUIRE(cuda_check_err_ == cudaSuccess); \
  } while (false)

// ---- device kernels: single thread evaluating one interval primitive ----
// Every primitive in cuinterval.cuh is __device__-only (it wraps CUDA
// rounding-mode intrinsics), so it can only be exercised from a kernel.

template<typename T>
__global__ void k_add_bounds(Bounds<T> a, Bounds<T> b, Bounds<T>* out)
{
  *out = add_bounds(a, b);
}

template<typename T>
__global__ void k_sub_bounds(Bounds<T> a, Bounds<T> b, Bounds<T>* out)
{
  *out = sub_bounds(a, b);
}

template<typename T>
__global__ void k_mul_bounds(Bounds<T> a, Bounds<T> b, Bounds<T>* out)
{
  *out = mul_bounds(a, b);
}

template<typename T>
__global__ void k_sqr_bound(Bounds<T> a, Bounds<T>* out)
{
  *out = sqr_bound(a);
}

template<typename T>
__global__ void k_scal_add_bound(Bounds<T> a, T x, Bounds<T>* out)
{
  *out = scal_add_bound(a, x);
}

template<typename T>
__global__ void k_scal_sub_bound_bs(Bounds<T> a, T x, Bounds<T>* out)
{
  *out = scal_sub_bound(a, x);
}

template<typename T>
__global__ void k_scal_sub_bound_sb(T x, Bounds<T> a, Bounds<T>* out)
{
  *out = scal_sub_bound(x, a);
}

template<typename T>
__global__ void k_scal_mul_bound(Bounds<T> a, T x, Bounds<T>* out)
{
  *out = scal_mul_bound(a, x);
}

// wraps a raw T[N] (or Bounds<T>[N]) so it can be passed through a kernel
// parameter list and still bind to sqr()/scale()/reduce_sum()'s flat
// Bounds<T>* / T* array parameters
template<typename T, std::size_t N>
struct coeffs
{
  T v[N];
};

template<typename T, std::size_t N>
struct bounds_array
{
  Bounds<T> v[N];
};

template<typename T, std::size_t N>
__global__ void k_sqr_interval(bounds_array<T, N> a, bounds_array<T, N>* out)
{
  sqr<T>(a.v, out->v, N);
}

template<typename T, std::size_t N>
__global__ void k_scale_interval(bounds_array<T, N> a,
                                 coeffs<T, N> c,
                                 bounds_array<T, N>* out)
{
  scale<T>(a.v, c.v, out->v, N);
}

template<typename T, std::size_t N>
__global__ void k_reduce_sum(bounds_array<T, N> a, Bounds<T>* out)
{
  *out = reduce_sum<T>(a.v, N);
}

// ---- host-side launch + copy-back ----

template<typename Out, typename Launch>
auto run_kernel(Launch launch) -> Out
{
  Out* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(Out)));
  launch(d_out);
  CUDA_CHECK(cudaGetLastError());
  Out h_out {};
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(Out), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  return h_out;
}

// ---- exact-reference rounding ----
//
// Every test interval below is built from small dyadic rationals (e.g.
// 2.5, 6.25, 9.5), so `long double` (64-bit mantissa on x86_64) represents
// every intermediate sum/product exactly, even a double x double product
// (which would otherwise need up to 106 bits to be exact). That gives a
// true mathematical result to round outward from, so tests can check for
// the tightest (0-ulp) correctly-rounded bound rather than only soundness.

template<typename T>
auto round_dn(long double exact) -> T
{
  T r = static_cast<T>(exact);
  while (static_cast<long double>(r) > exact) {
    r = std::nextafter(r, -std::numeric_limits<T>::infinity());
  }
  return r;
}

template<typename T>
auto round_up(long double exact) -> T
{
  T r = static_cast<T>(exact);
  while (static_cast<long double>(r) < exact) {
    r = std::nextafter(r, std::numeric_limits<T>::infinity());
  }
  return r;
}

template<typename T>
void require_bounds(Bounds<T> actual,
                    long double lo_exact,
                    long double up_exact)
{
  REQUIRE(actual.lower == round_dn<T>(lo_exact));
  REQUIRE(actual.upper == round_up<T>(up_exact));
}

// ---- Bounds<T> arithmetic ----

TEMPLATE_TEST_CASE("add_bounds covers sign combinations",
                   "[cuinterval][add]",
                   float,
                   double)
{
  using T = TestType;
  auto const B = [](long double lo, long double hi) -> Bounds<T>
  { return Bounds<T>(static_cast<T>(lo), static_cast<T>(hi)); };

  Bounds<T> const pos = B(2.5L, 6.25L);
  Bounds<T> const neg = B(-9.5L, -1.25L);
  Bounds<T> const straddling = B(-3.75L, 4.5L);

  SECTION("positive + positive")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_add_bounds<T><<<1, 1>>>(pos, pos, out); });
    require_bounds<T>(r, 5.0L, 12.5L);
  }

  SECTION("negative + negative")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_add_bounds<T><<<1, 1>>>(neg, neg, out); });
    require_bounds<T>(r, -19.0L, -2.5L);
  }

  SECTION("zero-straddling + zero-straddling")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_add_bounds<T><<<1, 1>>>(straddling, straddling, out); });
    require_bounds<T>(r, -7.5L, 9.0L);
  }

  SECTION("positive + negative crosses zero")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_add_bounds<T><<<1, 1>>>(pos, neg, out); });
    require_bounds<T>(r, -7.0L, 5.0L);
  }

  SECTION("positive + zero-straddling")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_add_bounds<T><<<1, 1>>>(pos, straddling, out); });
    require_bounds<T>(r, -1.25L, 10.75L);
  }

  SECTION("non-representable sum still rounds outward and tight")
  {
    Bounds<T> const a = B(0.1L, 0.1L);
    Bounds<T> const b = B(0.2L, 0.2L);
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_add_bounds<T><<<1, 1>>>(a, b, out); });
    long double const lo_exact =
        static_cast<long double>(a.lower) + static_cast<long double>(b.lower);
    long double const up_exact =
        static_cast<long double>(a.upper) + static_cast<long double>(b.upper);
    require_bounds<T>(r, lo_exact, up_exact);
    CHECK(r.lower < r.upper);
  }
}

TEMPLATE_TEST_CASE("sub_bounds covers sign combinations and argument order",
                   "[cuinterval][sub]",
                   float,
                   double)
{
  using T = TestType;
  auto const B = [](long double lo, long double hi) -> Bounds<T>
  { return Bounds<T>(static_cast<T>(lo), static_cast<T>(hi)); };

  Bounds<T> const pos = B(2.5L, 6.25L);
  Bounds<T> const neg = B(-9.5L, -1.25L);
  Bounds<T> const straddling = B(-3.75L, 4.5L);

  SECTION("positive - positive can straddle zero")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sub_bounds<T><<<1, 1>>>(pos, pos, out); });
    require_bounds<T>(r, -3.75L, 3.75L);
  }

  SECTION("negative - negative")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sub_bounds<T><<<1, 1>>>(neg, neg, out); });
    require_bounds<T>(r, -8.25L, 8.25L);
  }

  SECTION("zero-straddling - zero-straddling")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_sub_bounds<T><<<1, 1>>>(straddling, straddling, out); });
    require_bounds<T>(r, -8.25L, 8.25L);
  }

  SECTION("positive - negative and negative - positive are not symmetric")
  {
    auto const pos_minus_neg = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sub_bounds<T><<<1, 1>>>(pos, neg, out); });
    require_bounds<T>(pos_minus_neg, 3.75L, 15.75L);

    auto const neg_minus_pos = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sub_bounds<T><<<1, 1>>>(neg, pos, out); });
    require_bounds<T>(neg_minus_pos, -15.75L, -3.75L);
  }
}

TEMPLATE_TEST_CASE("mul_bounds covers sign combinations",
                   "[cuinterval][mul]",
                   float,
                   double)
{
  using T = TestType;
  auto const B = [](long double lo, long double hi) -> Bounds<T>
  { return Bounds<T>(static_cast<T>(lo), static_cast<T>(hi)); };

  Bounds<T> const pos = B(2.5L, 6.25L);
  Bounds<T> const neg = B(-9.5L, -1.25L);
  Bounds<T> const straddling = B(-3.75L, 4.5L);
  Bounds<T> const point_pos = B(3.125L, 3.125L);
  Bounds<T> const zero_pt = B(0.0L, 0.0L);

  SECTION("positive * positive")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_mul_bounds<T><<<1, 1>>>(pos, pos, out); });
    require_bounds<T>(r, 6.25L, 39.0625L);
  }

  SECTION("negative * negative is positive")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_mul_bounds<T><<<1, 1>>>(neg, neg, out); });
    require_bounds<T>(r, 1.5625L, 90.25L);
  }

  SECTION("positive * negative is negative")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_mul_bounds<T><<<1, 1>>>(pos, neg, out); });
    require_bounds<T>(r, -59.375L, -3.125L);
  }

  SECTION("zero-straddling * zero-straddling")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_mul_bounds<T><<<1, 1>>>(straddling, straddling, out); });
    require_bounds<T>(r, -16.875L, 20.25L);
  }

  SECTION("zero-straddling * positive")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_mul_bounds<T><<<1, 1>>>(straddling, pos, out); });
    require_bounds<T>(r, -23.4375L, 28.125L);
  }

  SECTION("zero-straddling * negative")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_mul_bounds<T><<<1, 1>>>(straddling, neg, out); });
    require_bounds<T>(r, -42.75L, 35.625L);
  }

  SECTION("degenerate point interval acts like a scalar")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_mul_bounds<T><<<1, 1>>>(point_pos, straddling, out); });
    require_bounds<T>(r, -11.71875L, 14.0625L);
  }

  SECTION("multiplying by a zero point interval collapses to zero")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_mul_bounds<T><<<1, 1>>>(zero_pt, straddling, out); });
    require_bounds<T>(r, 0.0L, 0.0L);
  }

  SECTION("non-representable product still rounds outward and tight")
  {
    Bounds<T> const a = B(0.1L, 0.1L);
    Bounds<T> const b = B(0.3L, 0.3L);
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_mul_bounds<T><<<1, 1>>>(a, b, out); });
    long double const lo_exact =
        static_cast<long double>(a.lower) * static_cast<long double>(b.lower);
    long double const up_exact =
        static_cast<long double>(a.upper) * static_cast<long double>(b.upper);
    require_bounds<T>(r, lo_exact, up_exact);
    CHECK(r.lower < r.upper);
  }
}

TEMPLATE_TEST_CASE(
    "sqr_bound picks the right branch across the sign of the interval",
    "[cuinterval][sqr]",
    float,
    double)
{
  using T = TestType;
  auto const B = [](long double lo, long double hi) -> Bounds<T>
  { return Bounds<T>(static_cast<T>(lo), static_cast<T>(hi)); };

  SECTION("wholly positive")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sqr_bound<T><<<1, 1>>>(B(2.5L, 6.25L), out); });
    require_bounds<T>(r, 6.25L, 39.0625L);
  }

  SECTION("wholly negative")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_sqr_bound<T><<<1, 1>>>(B(-9.5L, -1.25L), out); });
    require_bounds<T>(r, 1.5625L, 90.25L);
  }

  SECTION("straddles zero")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_sqr_bound<T><<<1, 1>>>(B(-3.75L, 4.5L), out); });
    require_bounds<T>(r, 0.0L, 20.25L);
  }

  SECTION("lower bound exactly zero takes the positive branch")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sqr_bound<T><<<1, 1>>>(B(0.0L, 5.5L), out); });
    require_bounds<T>(r, 0.0L, 30.25L);
  }

  SECTION("upper bound exactly zero takes the negative branch")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sqr_bound<T><<<1, 1>>>(B(-5.5L, 0.0L), out); });
    require_bounds<T>(r, 0.0L, 30.25L);
  }
}

TEMPLATE_TEST_CASE("scal_add_bound shifts an interval by a real scalar",
                   "[cuinterval][scal_add]",
                   float,
                   double)
{
  using T = TestType;
  auto const B = [](long double lo, long double hi) -> Bounds<T>
  { return Bounds<T>(static_cast<T>(lo), static_cast<T>(hi)); };

  SECTION("positive interval, positive scalar")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) {
          k_scal_add_bound<T>
              <<<1, 1>>>(B(2.5L, 6.25L), static_cast<T>(1.5L), out);
        });
    require_bounds<T>(r, 4.0L, 7.75L);
  }

  SECTION("negative interval, negative scalar")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        {
          k_scal_add_bound<T>
              <<<1, 1>>>(B(-9.5L, -1.25L), static_cast<T>(-2.5L), out);
        });
    require_bounds<T>(r, -12.0L, -3.75L);
  }

  SECTION("zero-straddling interval, zero scalar is the identity")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) {
          k_scal_add_bound<T>
              <<<1, 1>>>(B(-3.75L, 4.5L), static_cast<T>(0.0L), out);
        });
    require_bounds<T>(r, -3.75L, 4.5L);
  }
}

TEMPLATE_TEST_CASE("scal_sub_bound overloads are not accidentally symmetric",
                   "[cuinterval][scal_sub]",
                   float,
                   double)
{
  using T = TestType;
  Bounds<T> const a(static_cast<T>(2.0L), static_cast<T>(3.0L));
  T const x = static_cast<T>(5.0L);

  SECTION("interval - scalar")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_scal_sub_bound_bs<T><<<1, 1>>>(a, x, out); });
    require_bounds<T>(r, -3.0L, -2.0L);
  }

  SECTION("scalar - interval")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_scal_sub_bound_sb<T><<<1, 1>>>(x, a, out); });
    require_bounds<T>(r, 2.0L, 3.0L);
  }
}

TEMPLATE_TEST_CASE("scal_mul_bound branches on the sign of the scalar",
                   "[cuinterval][scal_mul]",
                   float,
                   double)
{
  using T = TestType;
  auto const B = [](long double lo, long double hi) -> Bounds<T>
  { return Bounds<T>(static_cast<T>(lo), static_cast<T>(hi)); };

  SECTION("positive scalar preserves orientation")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) {
          k_scal_mul_bound<T>
              <<<1, 1>>>(B(2.5L, 6.25L), static_cast<T>(2.0L), out);
        });
    require_bounds<T>(r, 5.0L, 12.5L);
  }

  SECTION("negative scalar flips orientation")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) {
          k_scal_mul_bound<T>
              <<<1, 1>>>(B(2.5L, 6.25L), static_cast<T>(-2.0L), out);
        });
    require_bounds<T>(r, -12.5L, -5.0L);
  }

  SECTION("zero scalar collapses a wide interval to a point at zero")
  {
    auto const r = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) {
          k_scal_mul_bound<T>
              <<<1, 1>>>(B(-3.75L, 4.5L), static_cast<T>(0.0L), out);
        });
    require_bounds<T>(r, 0.0L, 0.0L);
  }
}

// ---- flat Bounds<T> array-level ops ----

TEMPLATE_TEST_CASE("sqr applies sqr_bound elementwise across dimensions",
                   "[cuinterval][interval][sqr]",
                   float,
                   double)
{
  using T = TestType;
  constexpr std::size_t n = 3;

  bounds_array<T, n> a;
  a.v[0] = Bounds<T>(static_cast<T>(2.5L), static_cast<T>(6.25L));  // +
  a.v[1] = Bounds<T>(static_cast<T>(-9.5L), static_cast<T>(-1.25L));  // -
  a.v[2] = Bounds<T>(static_cast<T>(-3.75L), static_cast<T>(4.5L));  // 0

  auto const r = run_kernel<bounds_array<T, n>>(
      [&](bounds_array<T, n>* out) { k_sqr_interval<T, n><<<1, 1>>>(a, out); });

  require_bounds<T>(r.v[0], 6.25L, 39.0625L);
  require_bounds<T>(r.v[1], 1.5625L, 90.25L);
  require_bounds<T>(r.v[2], 0.0L, 20.25L);
}

TEMPLATE_TEST_CASE(
    "scale applies scal_mul_bound elementwise with per-dimension "
    "coefficients",
    "[cuinterval][interval][scale]",
    float,
    double
)
{
  using T = TestType;
  constexpr std::size_t n = 3;

  bounds_array<T, n> a;
  a.v[0] = Bounds<T>(static_cast<T>(2.5L), static_cast<T>(6.25L));
  a.v[1] = Bounds<T>(static_cast<T>(-9.5L), static_cast<T>(-1.25L));
  a.v[2] = Bounds<T>(static_cast<T>(-3.75L), static_cast<T>(4.5L));

  coeffs<T, n> c {
      {static_cast<T>(2.0L), static_cast<T>(-2.0L), static_cast<T>(0.0L)}};

  auto const r = run_kernel<bounds_array<T, n>>(
      [&](bounds_array<T, n>* out)
      { k_scale_interval<T, n><<<1, 1>>>(a, c, out); });

  require_bounds<T>(r.v[0], 5.0L, 12.5L);
  require_bounds<T>(r.v[1], 2.5L, 19.0L);
  require_bounds<T>(r.v[2], 0.0L, 0.0L);
}

TEMPLATE_TEST_CASE("reduce_sum folds mixed-sign dimensions into a single bound",
                   "[cuinterval][interval][reduce_sum]",
                   float,
                   double)
{
  using T = TestType;
  constexpr std::size_t n = 3;

  // integer bounds keep every partial sum exactly representable, so a
  // single long-double sum of all three dimensions is a valid reference
  // for the sequential add_bounds accumulation reduce_sum performs
  bounds_array<T, n> a;
  a.v[0] = Bounds<T>(static_cast<T>(2.0L), static_cast<T>(3.0L));
  a.v[1] = Bounds<T>(static_cast<T>(-5.0L), static_cast<T>(-1.0L));
  a.v[2] = Bounds<T>(static_cast<T>(-4.0L), static_cast<T>(6.0L));

  auto const r = run_kernel<Bounds<T>>(
      [&](Bounds<T>* out) { k_reduce_sum<T, n><<<1, 1>>>(a, out); });

  require_bounds<T>(r, -7.0L, 8.0L);
}

// ---- host (CpuRounding) path vs device (CudaRounding) path ----

TEMPLATE_TEST_CASE(
    "CpuRounding host path matches the CudaRounding device kernels bit-for-bit",
    "[cuinterval][host]",
    float,
    double)
{
  using T = TestType;
  auto const B = [](long double lo, long double hi) -> Bounds<T>
  { return Bounds<T>(static_cast<T>(lo), static_cast<T>(hi)); };

  Bounds<T> const pos = B(2.5L, 6.25L);
  Bounds<T> const neg = B(-9.5L, -1.25L);
  Bounds<T> const straddling = B(-3.75L, 4.5L);
  T const scalar = static_cast<T>(2.0L);

  SECTION("add_bounds")
  {
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_add_bounds<T><<<1, 1>>>(pos, straddling, out); });
    auto const host = add_bounds<T, CpuRounding>(pos, straddling);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }

  SECTION("sub_bounds")
  {
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sub_bounds<T><<<1, 1>>>(neg, pos, out); });
    auto const host = sub_bounds<T, CpuRounding>(neg, pos);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }

  SECTION("mul_bounds")
  {
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_mul_bounds<T><<<1, 1>>>(straddling, neg, out); });
    auto const host = mul_bounds<T, CpuRounding>(straddling, neg);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }

  SECTION("sqr_bound")
  {
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out) { k_sqr_bound<T><<<1, 1>>>(straddling, out); });
    auto const host = sqr_bound<T, CpuRounding>(straddling);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }

  SECTION("scal_add_bound")
  {
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_scal_add_bound<T><<<1, 1>>>(pos, scalar, out); });
    auto const host = scal_add_bound<T, CpuRounding>(pos, scalar);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }

  SECTION("scal_sub_bound (bounds - scalar)")
  {
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_scal_sub_bound_bs<T><<<1, 1>>>(pos, scalar, out); });
    auto const host = scal_sub_bound<T, CpuRounding>(pos, scalar);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }

  SECTION("scal_sub_bound (scalar - bounds)")
  {
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_scal_sub_bound_sb<T><<<1, 1>>>(scalar, pos, out); });
    auto const host = scal_sub_bound<T, CpuRounding>(scalar, pos);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }

  SECTION("scal_mul_bound")
  {
    T const neg_scalar = static_cast<T>(-2.0L);
    auto const device = run_kernel<Bounds<T>>(
        [&](Bounds<T>* out)
        { k_scal_mul_bound<T><<<1, 1>>>(straddling, neg_scalar, out); });
    auto const host = scal_mul_bound<T, CpuRounding>(straddling, neg_scalar);
    CHECK(host.lower == device.lower);
    CHECK(host.upper == device.upper);
  }
}
