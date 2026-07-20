#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "CuQCQPs/search.hpp"
#include "CuQCQPs/rosenbrock.cuh"
#include "cpu_rounding.hpp"

using namespace cuqcqps::rosenbrock;

// GPU arithmetic can contract multiply-adds differently than the host
// compiler, so results are compared with a relative tolerance rather than
// bit-for-bit. float accumulates more rounding noise than double over the
// DIMS-1 term sum, hence the looser tolerance.
template<typename T>
constexpr T rosenbrock_tolerance()
{
  return std::is_same_v<T, float> ? static_cast<T>(1e-3) : static_cast<T>(1e-9);
}

// cu::interval<T>'s arithmetic (operator+/-/*, sqr()) is __device__-only, so
// it can't be called from a host oracle. These reimplement just the composed
// ops the oracles below need, directly against CpuRounding (see
// cpu_rounding.hpp) -- the host-side counterpart to what the kernels compute
// with CudaRounding intrinsics.
namespace
{

template<typename T>
cu::interval<T> add_bounds(cu::interval<T> a, cu::interval<T> b)
{
  return cu::interval<T>(CpuRounding::add_rd(a.lb, b.lb),
                         CpuRounding::add_ru(a.ub, b.ub));
}

template<typename T>
cu::interval<T> sub_bounds(cu::interval<T> a, cu::interval<T> b)
{
  return cu::interval<T>(CpuRounding::sub_rd(a.lb, b.ub),
                         CpuRounding::sub_ru(a.ub, b.lb));
}

template<typename T>
cu::interval<T> sqr_bound(cu::interval<T> a)
{
  if (a.lb >= 0) {
    return cu::interval<T>(CpuRounding::mul_rd(a.lb, a.lb),
                           CpuRounding::mul_ru(a.ub, a.ub));
  } else if (a.ub <= 0) {
    return cu::interval<T>(CpuRounding::mul_rd(a.ub, a.ub),
                           CpuRounding::mul_ru(a.lb, a.lb));
  } else {
    return cu::interval<T>(0,
                           std::max(CpuRounding::mul_ru(a.lb, a.lb),
                                    CpuRounding::mul_ru(a.ub, a.ub)));
  }
}

template<typename T>
cu::interval<T> scal_sub_bound(cu::interval<T> a, T x)
{
  return cu::interval<T>(CpuRounding::sub_rd(a.lb, x),
                         CpuRounding::sub_ru(a.ub, x));
}

template<typename T>
cu::interval<T> scal_mul_bound(cu::interval<T> a, T x)
{
  if (x >= 0) {
    return cu::interval<T>(CpuRounding::mul_rd(a.lb, x),
                           CpuRounding::mul_ru(a.ub, x));
  } else {
    return cu::interval<T>(CpuRounding::mul_rd(a.ub, x),
                           CpuRounding::mul_ru(a.lb, x));
  }
}

}  // namespace

// Independent CPU implementation of interval_rosenbrock_kernel
// (rosenbrock.cuh), using the CpuRounding-backed composed ops above instead
// of the kernel's cu::interval device operators.
template<typename T>
cu::interval<T> interval_rosenbrock_cpu(const std::vector<cu::interval<T>>& iv)
{
  std::size_t const DIMS = iv.size();
  cu::interval<T> res(0);
  for (std::size_t i = 0; i < DIMS - 1; ++i) {
    cu::interval<T> xi = iv[i];
    cu::interval<T> xi1 = iv[i + 1];

    cu::interval<T> a = sub_bounds<T>(sqr_bound<T>(xi), xi1);
    cu::interval<T> b = scal_sub_bound<T>(xi, static_cast<T>(1));

    cu::interval<T> c = scal_mul_bound<T>(sqr_bound<T>(a), static_cast<T>(100));
    cu::interval<T> d = sqr_bound<T>(b);
    cu::interval<T> e = add_bounds<T>(c, d);
    res = add_bounds<T>(res, e);
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
  std::uniform_real_distribution<T> dist(static_cast<T>(-30),
                                         static_cast<T>(30));

  std::vector<std::vector<T>> points(N);
  for (auto& p : points) {
    p.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      p[d] = dist(rng);
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
  std::uniform_real_distribution<T> dist(static_cast<T>(-30),
                                         static_cast<T>(30));

  std::vector<std::vector<cu::interval<T>>> intervals(N);
  for (auto& iv : intervals) {
    iv.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv[d] = cu::interval<T>(std::min(a, b), std::max(a, b));
    }
  }

  auto const gpu_res = launch_interval_rosenbrock<T>(intervals);
  REQUIRE(gpu_res.size() == N);

  SECTION("matches the CPU rounding oracle bit-for-bit")
  {
    for (std::size_t i = 0; i < N; ++i) {
      auto const cpu_res = interval_rosenbrock_cpu<T>(intervals[i]);
      CAPTURE(i);
      CHECK(gpu_res[i].lb == cpu_res.lb);
      CHECK(gpu_res[i].ub == cpu_res.ub);
    }
  }

  SECTION("encloses sampled point evaluations")
  {
    std::uniform_real_distribution<T> unit(static_cast<T>(0),
                                           static_cast<T>(1));
    constexpr int SAMPLES_PER_INTERVAL = 20;
    for (std::size_t i = 0; i < N; ++i) {
      for (int s = 0; s < SAMPLES_PER_INTERVAL; ++s) {
        std::vector<T> p(DIMS);
        for (std::size_t d = 0; d < DIMS; ++d) {
          cu::interval<T> const& b = intervals[i][d];
          T const t = unit(rng);
          p[d] = b.lb + t * (b.ub - b.lb);
        }
        T const value = rosenbrock<T>(p);
        CAPTURE(i, s);
        CHECK(value >= gpu_res[i].lb);
        CHECK(value <= gpu_res[i].ub);
      }
    }
  }
}

// Independent CPU implementation of sample_rosenbrock_kernel (rosenbrock.cuh),
// mirroring its tid -> CycleContext -> sample-point partitioning verbatim
// (see make_cycle_context/get_bounds in partition.cuh) with the CpuRounding-backed
// composed ops above instead of the kernel's cu::interval device operators.
// The partitioning scheme itself is fixed; only the
// CycleSize/PartitionNum/SamplePoints template values vary between runs, so
// this oracle can be reused as-is.
template<typename T,
         std::size_t CycleSize,
         std::size_t PartitionNum,
         std::size_t SamplePoints>
T sample_rosenbrock_cpu(const std::vector<cu::interval<T>>& iv,
                        std::size_t cycle_start)
{
  std::size_t const dims = iv.size();
  std::size_t const num_threads = ipow<PartitionNum, CycleSize>();

  T lub = std::numeric_limits<T>::max();

  for (std::size_t tid = 0; tid < num_threads; ++tid) {
    std::size_t part[CycleSize];
    std::size_t idx = tid;
    for (std::size_t j = 0; j < CycleSize; ++j) {
      part[j] = idx % PartitionNum;
      idx /= PartitionNum;
    }

    auto const bounds_for_dim = [&](std::size_t dim) -> cu::interval<T>
    {
      if (dim >= cycle_start && dim < cycle_start + CycleSize) {
        std::size_t const i = dim - cycle_start;
        T const width =
            (iv[dim].ub - iv[dim].lb) / static_cast<T>(PartitionNum);
        return cu::interval<T>(
            iv[dim].lb + width * static_cast<T>(part[i]),
            iv[dim].lb + width * static_cast<T>(part[i] + 1));
      }
      return iv[dim];
    };

    T ub = std::numeric_limits<T>::max();
    for (std::size_t i = 0; i < SamplePoints; ++i) {
      cu::interval<T> res(0);
      for (std::size_t j = 0; j + 1 < dims; ++j) {
        cu::interval<T> const int_xj = bounds_for_dim(j);
        cu::interval<T> const int_xj1 = bounds_for_dim(j + 1);

        cu::interval<T> const xj(int_xj.lb
                                 + static_cast<T>(i) * (int_xj.ub - int_xj.lb)
                                     / static_cast<T>(SamplePoints - 1));
        cu::interval<T> const xj1(int_xj1.lb
                                  + static_cast<T>(i)
                                      * (int_xj1.ub - int_xj1.lb)
                                      / static_cast<T>(SamplePoints - 1));

        cu::interval<T> a = sub_bounds<T>(sqr_bound<T>(xj), xj1);
        cu::interval<T> b = scal_sub_bound<T>(xj, static_cast<T>(1));

        cu::interval<T> c =
            scal_mul_bound<T>(sqr_bound<T>(a), static_cast<T>(100));
        cu::interval<T> d = sqr_bound<T>(b);
        cu::interval<T> e = add_bounds<T>(c, d);
        res = add_bounds<T>(res, e);
      }
      ub = std::min(ub, res.ub);
    }
    lub = std::min(lub, ub);
  }
  return lub;
}

TEMPLATE_TEST_CASE(
    "sample_rosenbrock_kernel matches the CPU partitioning oracle",
    "[rosenbrock][sample]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 6;
  constexpr std::size_t CYCLE_SIZE = 2;
  constexpr std::size_t PARTITION_NUM = 3;
  constexpr std::size_t SAMPLE_POINTS = 10;
  constexpr std::size_t N = 8;

  std::mt19937 rng(24601);
  std::uniform_real_distribution<T> dist(static_cast<T>(-30),
                                         static_cast<T>(30));

  std::vector<std::vector<cu::interval<T>>> intervals(N);
  for (auto& iv : intervals) {
    iv.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv[d] = cu::interval<T>(std::min(a, b), std::max(a, b));
    }
  }

  auto const tol = rosenbrock_tolerance<T>();

  // cycle_start = 0 covers the leading dims; DIMS - CYCLE_SIZE exercises a
  // non-zero offset so the cycleStart window logic in get_bounds is checked
  // at both ends.
  for (std::size_t const cycle_start : {std::size_t {0}, DIMS - CYCLE_SIZE}) {
    for (std::size_t i = 0; i < N; ++i) {
      T const gpu_res =
          launch_sample_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS>(
              intervals[i], cycle_start);
      T const cpu_res =
          sample_rosenbrock_cpu<T, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS>(
              intervals[i], cycle_start);
      CAPTURE(i, cycle_start);
      REQUIRE(gpu_res == Catch::Approx(cpu_res).epsilon(tol));
    }
  }
}

TEMPLATE_TEST_CASE(
    "sample_rosenbrock_kernel finds the exact global minimum when it lands on "
    "the sample grid",
    "[rosenbrock][sample]",
    float,
    double)
{
  using T = TestType;

  // A single, unpartitioned thread (CycleSize=1, PartitionNum=1) sampling the
  // diagonal of [-1, 1]^2 at 10 evenly spaced points lands exactly on
  // (1, 1) at the last sample (i=9 of 9, fraction 9/9=1 -> x=upper=1), which
  // is Rosenbrock's known global minimum (value 0). Every quantity involved
  // (-1, 1, 2, 9, 100) is exactly representable, so this holds bit-for-bit,
  // not just approximately.
  std::vector<cu::interval<T>> iv = {
      cu::interval<T>(static_cast<T>(-1), static_cast<T>(1)),
      cu::interval<T>(static_cast<T>(-1), static_cast<T>(1))};

  T const lub = launch_sample_rosenbrock<T, 1, 1, 10>(iv, 0);
  REQUIRE(lub == static_cast<T>(0));
}

// Independent CPU implementation of bound_rosenbrock_kernel (rosenbrock.cuh),
// mirroring its tid -> CycleContext -> subinterval partitioning verbatim (see
// sample_rosenbrock_cpu above), but evaluating the full interval enclosure
// per subinterval (like interval_rosenbrock_cpu) instead of sampling points,
// then comparing the enclosure's lower bound against gub with the
// CpuRounding-backed composed ops above.
//
// NOTE: this mirrors the kernel's partitioning and arithmetic chain rather
// than deriving the answer a different way, so it cannot catch a bug present
// in both the kernel and this oracle (e.g. a wrong exponent shared by both,
// or an off-by-one in the partition math both copy). It's still useful as a
// tight regression pin. The tests further below (witness-point soundness and
// gub-monotonicity) are structurally independent of this oracle and of each
// other, and exist specifically to catch that class of correlated bug.
template<typename T, std::size_t CycleSize, std::size_t PartitionNum>
std::array<bool, ipow<PartitionNum, CycleSize>()> bound_rosenbrock_cpu(
    const std::vector<cu::interval<T>>& iv,
    T gub,
    std::size_t cycle_start,
    std::array<T, ipow<PartitionNum, CycleSize>()>& lb_out)
{
  std::size_t const dims = iv.size();
  std::size_t const num_threads = ipow<PartitionNum, CycleSize>();

  std::array<bool, ipow<PartitionNum, CycleSize>()> result {};

  for (std::size_t tid = 0; tid < num_threads; ++tid) {
    std::size_t part[CycleSize];
    std::size_t idx = tid;
    for (std::size_t j = 0; j < CycleSize; ++j) {
      part[j] = idx % PartitionNum;
      idx /= PartitionNum;
    }

    auto const bounds_for_dim = [&](std::size_t dim) -> cu::interval<T>
    {
      if (dim >= cycle_start && dim < cycle_start + CycleSize) {
        std::size_t const i = dim - cycle_start;
        T const width =
            (iv[dim].ub - iv[dim].lb) / static_cast<T>(PartitionNum);
        return cu::interval<T>(
            iv[dim].lb + width * static_cast<T>(part[i]),
            iv[dim].lb + width * static_cast<T>(part[i] + 1));
      }
      return iv[dim];
    };

    cu::interval<T> res(0);
    for (std::size_t i = 0; i + 1 < dims; ++i) {
      cu::interval<T> const xi = bounds_for_dim(i);
      cu::interval<T> const xi1 = bounds_for_dim(i + 1);

      cu::interval<T> a = sub_bounds<T>(sqr_bound<T>(xi), xi1);
      cu::interval<T> b = scal_sub_bound<T>(xi, static_cast<T>(1));

      cu::interval<T> c =
          scal_mul_bound<T>(sqr_bound<T>(a), static_cast<T>(100));
      cu::interval<T> d = sqr_bound<T>(b);
      cu::interval<T> e = add_bounds<T>(c, d);
      res = add_bounds<T>(res, e);
    }
    result[tid] = (res.lb > gub);
    lb_out[tid] = res.lb;
  }
  return result;
}

TEMPLATE_TEST_CASE(
    "bound_rosenbrock_kernel matches the CPU partitioning oracle",
    "[rosenbrock][bound]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 6;
  constexpr std::size_t CYCLE_SIZE = 2;
  constexpr std::size_t PARTITION_NUM = 3;
  constexpr std::size_t N = 8;

  std::mt19937 rng(112358);
  std::uniform_real_distribution<T> dist(static_cast<T>(-30),
                                         static_cast<T>(30));
  // Chosen to be within the range of enclosure lower bounds these intervals
  // actually produce, so both prune (true) and keep (false) outcomes occur.
  std::uniform_real_distribution<T> gub_dist(static_cast<T>(0),
                                             static_cast<T>(5e7));

  std::vector<std::vector<cu::interval<T>>> intervals(N);
  for (auto& iv : intervals) {
    iv.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv[d] = cu::interval<T>(std::min(a, b), std::max(a, b));
    }
  }

  auto const tol = rosenbrock_tolerance<T>();

  // cycle_start = 0 covers the leading dims; DIMS - CYCLE_SIZE exercises a
  // non-zero offset so the cycleStart window logic in get_bounds is checked
  // at both ends.
  for (std::size_t const cycle_start : {std::size_t {0}, DIMS - CYCLE_SIZE}) {
    for (std::size_t i = 0; i < N; ++i) {
      T const gub = gub_dist(rng);

      std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> gpu_res {};
      std::array<T, ipow<PARTITION_NUM, CYCLE_SIZE>()> gpu_lb {};
      launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(
          intervals[i], gub, cycle_start, gpu_res, gpu_lb);

      std::array<T, ipow<PARTITION_NUM, CYCLE_SIZE>()> cpu_lb {};
      auto const cpu_res = bound_rosenbrock_cpu<T, CYCLE_SIZE, PARTITION_NUM>(
          intervals[i], gub, cycle_start, cpu_lb);

      CAPTURE(i, cycle_start, gub);
      for (std::size_t tid = 0; tid < gpu_res.size(); ++tid) {
        CAPTURE(tid);
        CHECK(gpu_res[tid] == cpu_res[tid]);
        // Partitioning arithmetic (get_bounds) uses plain FP ops rather than
        // the directed-rounding Cpu/CudaRounding intrinsics, so GPU/CPU can
        // differ by FMA contraction; compare with tolerance like the
        // sample_rosenbrock oracle above, not bit-for-bit.
        CHECK(gpu_lb[tid] == Catch::Approx(cpu_lb[tid]).epsilon(tol));
      }
    }
  }
}

TEMPLATE_TEST_CASE(
    "bound_rosenbrock_kernel prunes every region below the true minimum and "
    "none above the " "maximum achievable enclosure",
    "[rosenbrock][bound]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 6;
  constexpr std::size_t CYCLE_SIZE = 2;
  constexpr std::size_t PARTITION_NUM = 3;

  std::vector<cu::interval<T>> iv(DIMS);
  for (auto& b : iv) {
    b = cu::interval<T>(static_cast<T>(-30), static_cast<T>(30));
  }

  // Rosenbrock is a sum of squares, so every subinterval's enclosure lower
  // bound is >= 0; a gub below that prunes every subinterval.
  {
    std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> res {};
    std::array<T, ipow<PARTITION_NUM, CYCLE_SIZE>()> lb {};
    launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(
        iv, static_cast<T>(-1), 0, res, lb);
    REQUIRE(std::all_of(res.begin(), res.end(), [](bool p) { return p; }));
    for (std::size_t tid = 0; tid < res.size(); ++tid) {
      CHECK(lb[tid] >= static_cast<T>(0));
    }
  }

  // A gub far above anything achievable over [-30, 30]^DIMS keeps every
  // subinterval.
  {
    std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> res {};
    std::array<T, ipow<PARTITION_NUM, CYCLE_SIZE>()> lb {};
    launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(
        iv, static_cast<T>(1e15), 0, res, lb);
    REQUIRE(std::none_of(res.begin(), res.end(), [](bool p) { return p; }));
  }
}

TEMPLATE_TEST_CASE(
    "bound_rosenbrock_kernel never prunes a subinterval that contains a point "
    "at or below gub",
    "[rosenbrock][bound]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 6;
  constexpr std::size_t CYCLE_SIZE = 2;
  constexpr std::size_t PARTITION_NUM = 3;
  constexpr std::size_t N = 8;
  constexpr int SAMPLES_PER_BOX = 20;

  std::mt19937 rng(271828);
  std::uniform_real_distribution<T> dist(static_cast<T>(-30),
                                         static_cast<T>(30));
  std::uniform_real_distribution<T> gub_dist(static_cast<T>(0),
                                             static_cast<T>(5e7));
  std::uniform_real_distribution<T> unit(static_cast<T>(0), static_cast<T>(1));

  std::vector<std::vector<cu::interval<T>>> intervals(N);
  for (auto& iv : intervals) {
    iv.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv[d] = cu::interval<T>(std::min(a, b), std::max(a, b));
    }
  }

  // Interval soundness guarantees res.lb <= f(x) for every x in the box, so
  // if the kernel marks a box pruned (res.lb > gub), no point inside that box
  // can score <= gub. This checks that implication directly with the plain
  // scalar rosenbrock<T>() (already validated against the CPU rounding
  // oracle in the rosenbrock_kernel test above), not with interval
  // arithmetic, so it doesn't share code with bound_rosenbrock_kernel or
  // with the mirrored oracle above -- a bug common to both of those would
  // still be caught here via a witness point.
  for (std::size_t const cycle_start : {std::size_t {0}, DIMS - CYCLE_SIZE}) {
    for (std::size_t i = 0; i < N; ++i) {
      T const gub = gub_dist(rng);

      std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> pruned {};
      std::array<T, ipow<PARTITION_NUM, CYCLE_SIZE>()> lb {};
      launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(
          intervals[i], gub, cycle_start, pruned, lb);

      for (std::size_t tid = 0; tid < pruned.size(); ++tid) {
        CAPTURE(i, cycle_start, gub, tid);
        CHECK(pruned[tid] == (lb[tid] > gub));

        // Reconstruct this tid's box directly from the parent interval and
        // the mixed-radix decomposition of tid, without calling get_bounds/
        // CycleContext (device-only, and shared with the kernel) or reusing
        // bound_rosenbrock_cpu's bounds_for_dim lambda above.
        std::vector<cu::interval<T>> box = intervals[i];
        std::size_t idx = tid;
        for (std::size_t d = cycle_start; d < cycle_start + CYCLE_SIZE; ++d) {
          std::size_t const part = idx % PARTITION_NUM;
          idx /= PARTITION_NUM;
          T const width = (intervals[i][d].ub - intervals[i][d].lb)
              / static_cast<T>(PARTITION_NUM);
          box[d] = cu::interval<T>(
              intervals[i][d].lb + width * static_cast<T>(part),
              intervals[i][d].lb + width * static_cast<T>(part + 1));
        }

        for (int s = 0; s < SAMPLES_PER_BOX; ++s) {
          std::vector<T> p(DIMS);
          for (std::size_t d = 0; d < DIMS; ++d) {
            T const t = unit(rng);
            p[d] = box[d].lb + t * (box[d].ub - box[d].lb);
          }
          T const value = rosenbrock<T>(p);
          if (value <= gub) {
            CAPTURE(i, cycle_start, gub, tid, s, value);
            CHECK_FALSE(pruned[tid]);
          }
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE(
    "bound_rosenbrock_kernel prunes a monotonically shrinking set of "
    "subintervals as gub increases",
    "[rosenbrock][bound]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 6;
  constexpr std::size_t CYCLE_SIZE = 2;
  constexpr std::size_t PARTITION_NUM = 3;
  constexpr std::size_t N = 8;

  std::mt19937 rng(31415);
  std::uniform_real_distribution<T> dist(static_cast<T>(-30),
                                         static_cast<T>(30));

  std::vector<std::vector<cu::interval<T>>> intervals(N);
  for (auto& iv : intervals) {
    iv.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv[d] = cu::interval<T>(std::min(a, b), std::max(a, b));
    }
  }

  // prune is (res.lb > gub) against a fixed per-box res.lb, so raising gub
  // can only turn prunes off, never on: whatever tid is pruned at some gub
  // must also be pruned at every smaller gub. This holds no matter what
  // res.lb actually evaluates to, so unlike every other test in this file it
  // needs no oracle at all -- just two runs of the launcher itself -- and
  // catches e.g. a flipped comparison direction that the mirrored oracle
  // above would reproduce identically and so could never catch.
  std::array<T, 4> const gubs = {static_cast<T>(0),
                                 static_cast<T>(100),
                                 static_cast<T>(1e5),
                                 static_cast<T>(1e10)};

  for (std::size_t i = 0; i < N; ++i) {
    std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> prev_pruned {};
    prev_pruned.fill(true);
    std::array<T, ipow<PARTITION_NUM, CYCLE_SIZE>()> first_lb {};
    bool have_first_lb = false;
    for (T const gub : gubs) {
      std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> pruned {};
      std::array<T, ipow<PARTITION_NUM, CYCLE_SIZE>()> lb {};
      launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(
          intervals[i], gub, 0, pruned, lb);

      for (std::size_t tid = 0; tid < pruned.size(); ++tid) {
        if (pruned[tid]) {
          CAPTURE(i, gub, tid);
          CHECK(prev_pruned[tid]);
        }
        // The enclosure's lower bound is a fixed property of the box, so it
        // must not change as gub varies across runs.
        if (have_first_lb) {
          CAPTURE(i, gub, tid);
          CHECK(lb[tid] == first_lb[tid]);
        }
      }
      prev_pruned = pruned;
      first_lb = lb;
      have_first_lb = true;
    }
  }
}
