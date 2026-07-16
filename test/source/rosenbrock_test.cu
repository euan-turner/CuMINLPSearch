#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "CuQCQPs/interval.hpp"
#include "CuQCQPs/rosenbrock.cuh"
#include "cpu_rounding.hpp"

using namespace cuqcqps::interval;
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

// Independent CPU implementation of sample_rosenbrock_kernel (rosenbrock.cuh),
// mirroring its tid -> CycleContext -> sample-point partitioning verbatim
// (see make_cycle_context/get_bounds in opt.cuh) with CpuRounding instead of
// the kernel's CudaRounding. The partitioning scheme itself is fixed; only
// the CycleSize/PartitionNum/SamplePoints template values vary between runs,
// so this oracle can be reused as-is.
template<typename T, std::size_t CycleSize, std::size_t PartitionNum, std::size_t SamplePoints>
T sample_rosenbrock_cpu(const Interval<T>& iv, std::size_t cycle_start)
{
  std::size_t const dims = iv.bounds.size();
  std::size_t const num_threads = ipow<PartitionNum, CycleSize>();

  T lub = std::numeric_limits<T>::max();

  for (std::size_t tid = 0; tid < num_threads; ++tid) {
    std::size_t part[CycleSize];
    std::size_t idx = tid;
    for (std::size_t j = 0; j < CycleSize; ++j) {
      part[j] = idx % PartitionNum;
      idx /= PartitionNum;
    }

    auto const bounds_for_dim = [&](std::size_t dim) -> Bounds<T> {
      if (dim >= cycle_start && dim < cycle_start + CycleSize) {
        std::size_t const i = dim - cycle_start;
        T const width = (iv.bounds[dim].upper - iv.bounds[dim].lower) / static_cast<T>(PartitionNum);
        return Bounds<T>(iv.bounds[dim].lower + width * static_cast<T>(part[i]),
                          iv.bounds[dim].lower + width * static_cast<T>(part[i] + 1));
      }
      return iv.bounds[dim];
    };

    T ub = std::numeric_limits<T>::max();
    for (std::size_t i = 0; i < SamplePoints; ++i) {
      Bounds<T> res(0);
      for (std::size_t j = 0; j + 1 < dims; ++j) {
        Bounds<T> const int_xj = bounds_for_dim(j);
        Bounds<T> const int_xj1 = bounds_for_dim(j + 1);

        Bounds<T> const xj(
            int_xj.lower + static_cast<T>(i) * (int_xj.upper - int_xj.lower) / static_cast<T>(SamplePoints - 1));
        Bounds<T> const xj1(
            int_xj1.lower + static_cast<T>(i) * (int_xj1.upper - int_xj1.lower) / static_cast<T>(SamplePoints - 1));

        Bounds<T> a = sub_bounds<T, CpuRounding>(sqr_bound<T, CpuRounding>(xj), xj1);
        Bounds<T> b = scal_sub_bound<T, CpuRounding>(xj, static_cast<T>(1));

        Bounds<T> c = scal_mul_bound<T, CpuRounding>(sqr_bound<T, CpuRounding>(a), static_cast<T>(100));
        Bounds<T> d = sqr_bound<T, CpuRounding>(b);
        Bounds<T> e = add_bounds<T, CpuRounding>(c, d);
        res = add_bounds<T, CpuRounding>(res, e);
      }
      ub = std::min(ub, res.upper);
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

  auto const tol = rosenbrock_tolerance<T>();

  // cycle_start = 0 covers the leading dims; DIMS - CYCLE_SIZE exercises a
  // non-zero offset so the cycleStart window logic in get_bounds is checked
  // at both ends.
  for (std::size_t const cycle_start : {std::size_t{0}, DIMS - CYCLE_SIZE}) {
    for (std::size_t i = 0; i < N; ++i) {
      T const gpu_res = launch_sample_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS>(
          intervals[i], cycle_start);
      T const cpu_res = sample_rosenbrock_cpu<T, CYCLE_SIZE, PARTITION_NUM, SAMPLE_POINTS>(
          intervals[i], cycle_start);
      CAPTURE(i, cycle_start);
      REQUIRE(gpu_res == Catch::Approx(cpu_res).epsilon(tol));
    }
  }
}

TEMPLATE_TEST_CASE(
    "sample_rosenbrock_kernel finds the exact global minimum when it lands on the sample grid",
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
  Interval<T> iv;
  iv.bounds = {Bounds<T>(static_cast<T>(-1), static_cast<T>(1)), Bounds<T>(static_cast<T>(-1), static_cast<T>(1))};

  T const lub = launch_sample_rosenbrock<T, 1, 1, 10>(iv, 0);
  REQUIRE(lub == static_cast<T>(0));
}

// Independent CPU implementation of bound_rosenbrock_kernel (rosenbrock.cuh),
// mirroring its tid -> CycleContext -> subinterval partitioning verbatim (see
// sample_rosenbrock_cpu above), but evaluating the full interval enclosure
// per subinterval (like interval_rosenbrock_cpu) instead of sampling points,
// then comparing the enclosure's lower bound against gub with CpuRounding
// instead of the kernel's CudaRounding.
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
    const Interval<T>& iv, T gub, std::size_t cycle_start)
{
  std::size_t const dims = iv.bounds.size();
  std::size_t const num_threads = ipow<PartitionNum, CycleSize>();

  std::array<bool, ipow<PartitionNum, CycleSize>()> result{};

  for (std::size_t tid = 0; tid < num_threads; ++tid) {
    std::size_t part[CycleSize];
    std::size_t idx = tid;
    for (std::size_t j = 0; j < CycleSize; ++j) {
      part[j] = idx % PartitionNum;
      idx /= PartitionNum;
    }

    auto const bounds_for_dim = [&](std::size_t dim) -> Bounds<T> {
      if (dim >= cycle_start && dim < cycle_start + CycleSize) {
        std::size_t const i = dim - cycle_start;
        T const width = (iv.bounds[dim].upper - iv.bounds[dim].lower) / static_cast<T>(PartitionNum);
        return Bounds<T>(iv.bounds[dim].lower + width * static_cast<T>(part[i]),
                          iv.bounds[dim].lower + width * static_cast<T>(part[i] + 1));
      }
      return iv.bounds[dim];
    };

    Bounds<T> res(0);
    for (std::size_t i = 0; i + 1 < dims; ++i) {
      Bounds<T> const xi = bounds_for_dim(i);
      Bounds<T> const xi1 = bounds_for_dim(i + 1);

      Bounds<T> a = sub_bounds<T, CpuRounding>(sqr_bound<T, CpuRounding>(xi), xi1);
      Bounds<T> b = scal_sub_bound<T, CpuRounding>(xi, static_cast<T>(1));

      Bounds<T> c = scal_mul_bound<T, CpuRounding>(sqr_bound<T, CpuRounding>(a), static_cast<T>(100));
      Bounds<T> d = sqr_bound<T, CpuRounding>(b);
      Bounds<T> e = add_bounds<T, CpuRounding>(c, d);
      res = add_bounds<T, CpuRounding>(res, e);
    }
    result[tid] = (res.lower > gub);
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
  std::uniform_real_distribution<T> dist(static_cast<T>(-30), static_cast<T>(30));
  // Chosen to be within the range of enclosure lower bounds these intervals
  // actually produce, so both prune (true) and keep (false) outcomes occur.
  std::uniform_real_distribution<T> gub_dist(static_cast<T>(0), static_cast<T>(5e7));

  std::vector<Interval<T>> intervals(N);
  for (auto& iv : intervals) {
    iv.bounds.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv.bounds[d] = Bounds<T>(std::min(a, b), std::max(a, b));
    }
  }

  // cycle_start = 0 covers the leading dims; DIMS - CYCLE_SIZE exercises a
  // non-zero offset so the cycleStart window logic in get_bounds is checked
  // at both ends.
  for (std::size_t const cycle_start : {std::size_t{0}, DIMS - CYCLE_SIZE}) {
    for (std::size_t i = 0; i < N; ++i) {
      T const gub = gub_dist(rng);

      std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> gpu_res{};
      launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(intervals[i], gub, cycle_start, gpu_res);

      auto const cpu_res = bound_rosenbrock_cpu<T, CYCLE_SIZE, PARTITION_NUM>(intervals[i], gub, cycle_start);

      CAPTURE(i, cycle_start, gub);
      for (std::size_t tid = 0; tid < gpu_res.size(); ++tid) {
        CAPTURE(tid);
        CHECK(gpu_res[tid] == cpu_res[tid]);
      }
    }
  }
}

TEMPLATE_TEST_CASE(
    "bound_rosenbrock_kernel prunes every region below the true minimum and none above the "
    "maximum achievable enclosure",
    "[rosenbrock][bound]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t DIMS = 6;
  constexpr std::size_t CYCLE_SIZE = 2;
  constexpr std::size_t PARTITION_NUM = 3;

  Interval<T> iv;
  iv.bounds.resize(DIMS);
  for (auto& b : iv.bounds) {
    b = Bounds<T>(static_cast<T>(-30), static_cast<T>(30));
  }

  // Rosenbrock is a sum of squares, so every subinterval's enclosure lower
  // bound is >= 0; a gub below that prunes every subinterval.
  {
    std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> res{};
    launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(iv, static_cast<T>(-1), 0, res);
    REQUIRE(std::all_of(res.begin(), res.end(), [](bool p) { return p; }));
  }

  // A gub far above anything achievable over [-30, 30]^DIMS keeps every
  // subinterval.
  {
    std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> res{};
    launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(iv, static_cast<T>(1e15), 0, res);
    REQUIRE(std::none_of(res.begin(), res.end(), [](bool p) { return p; }));
  }
}

TEMPLATE_TEST_CASE(
    "bound_rosenbrock_kernel never prunes a subinterval that contains a point at or below gub",
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
  std::uniform_real_distribution<T> dist(static_cast<T>(-30), static_cast<T>(30));
  std::uniform_real_distribution<T> gub_dist(static_cast<T>(0), static_cast<T>(5e7));
  std::uniform_real_distribution<T> unit(static_cast<T>(0), static_cast<T>(1));

  std::vector<Interval<T>> intervals(N);
  for (auto& iv : intervals) {
    iv.bounds.resize(DIMS);
    for (std::size_t d = 0; d < DIMS; ++d) {
      T const a = dist(rng);
      T const b = dist(rng);
      iv.bounds[d] = Bounds<T>(std::min(a, b), std::max(a, b));
    }
  }

  // Interval soundness guarantees res.lower <= f(x) for every x in the box,
  // so if the kernel marks a box pruned (res.lower > gub), no point inside
  // that box can score <= gub. This checks that implication directly with
  // the plain scalar rosenbrock<T>() (already validated against the CPU
  // rounding oracle in the rosenbrock_kernel test above), not with interval
  // arithmetic, so it doesn't share code with bound_rosenbrock_kernel or
  // with the mirrored oracle above -- a bug common to both of those would
  // still be caught here via a witness point.
  for (std::size_t const cycle_start : {std::size_t{0}, DIMS - CYCLE_SIZE}) {
    for (std::size_t i = 0; i < N; ++i) {
      T const gub = gub_dist(rng);

      std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> pruned{};
      launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(intervals[i], gub, cycle_start, pruned);

      for (std::size_t tid = 0; tid < pruned.size(); ++tid) {
        // Reconstruct this tid's box directly from the parent interval and
        // the mixed-radix decomposition of tid, without calling get_bounds/
        // CycleContext (device-only, and shared with the kernel) or reusing
        // bound_rosenbrock_cpu's bounds_for_dim lambda above.
        Interval<T> box;
        box.bounds = intervals[i].bounds;
        std::size_t idx = tid;
        for (std::size_t d = cycle_start; d < cycle_start + CYCLE_SIZE; ++d) {
          std::size_t const part = idx % PARTITION_NUM;
          idx /= PARTITION_NUM;
          T const width =
              (intervals[i].bounds[d].upper - intervals[i].bounds[d].lower) / static_cast<T>(PARTITION_NUM);
          box.bounds[d] = Bounds<T>(intervals[i].bounds[d].lower + width * static_cast<T>(part),
                                     intervals[i].bounds[d].lower + width * static_cast<T>(part + 1));
        }

        for (int s = 0; s < SAMPLES_PER_BOX; ++s) {
          Point<T> p;
          p.elems.resize(DIMS);
          for (std::size_t d = 0; d < DIMS; ++d) {
            T const t = unit(rng);
            p.elems[d] = box.bounds[d].lower + t * (box.bounds[d].upper - box.bounds[d].lower);
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
    "bound_rosenbrock_kernel prunes a monotonically shrinking set of subintervals as gub increases",
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

  // prune is (res.lower > gub) against a fixed per-box res.lower, so raising
  // gub can only turn prunes off, never on: whatever tid is pruned at some
  // gub must also be pruned at every smaller gub. This holds no matter what
  // res.lower actually evaluates to, so unlike every other test in this
  // file it needs no oracle at all -- just two runs of the launcher itself
  // -- and catches e.g. a flipped comparison direction that the mirrored
  // oracle above would reproduce identically and so could never catch.
  std::array<T, 4> const gubs = {static_cast<T>(0), static_cast<T>(100), static_cast<T>(1e5), static_cast<T>(1e10)};

  for (std::size_t i = 0; i < N; ++i) {
    std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> prev_pruned{};
    prev_pruned.fill(true);
    for (T const gub : gubs) {
      std::array<bool, ipow<PARTITION_NUM, CYCLE_SIZE>()> pruned{};
      launch_bound_rosenbrock<T, CYCLE_SIZE, PARTITION_NUM>(intervals[i], gub, 0, pruned);

      for (std::size_t tid = 0; tid < pruned.size(); ++tid) {
        if (pruned[tid]) {
          CAPTURE(i, gub, tid);
          CHECK(prev_pruned[tid]);
        }
      }
      prev_pruned = pruned;
    }
  }
}
