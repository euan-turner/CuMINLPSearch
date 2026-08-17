// design/REFINEMENT_STUDY.md stage 4: the five-layer distribution summary
// that answers Q2 -- are the per-subregion bounds all alike, or are the
// extremes isolated outliers?
//
// Tested against planted arrays rather than solver output, deliberately. This
// is ordinary statistics and it should be pinned as such: an array with one
// outlier and an array with a cluster of ten must produce visibly different
// numbers, and that is checkable without a GPU, a problem, or a launch.
//
// Host-only: no CUDA, no device required.

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cuminlp/study/distribution.hpp"

using cuminlp::study::concentration;
using cuminlp::study::Distribution;
using cuminlp::study::kIsolationRanks;
using cuminlp::study::kQuantileCount;
using cuminlp::study::summarise;

namespace
{

/// Exact equality without tripping -Werror=float-equal, the same idiom
/// aggregate_backend_test.cu uses. These really are exact comparisons: every
/// value checked below is either a planted constant or an order statistic
/// copied straight out of the input, so anything but exact agreement is a
/// real disagreement rather than rounding.
bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

/// A featureless uniform bulk: 0, 1, ..., n-1. Has no outliers by
/// construction, so it is the control against which the planted cases below
/// must look different.
std::vector<double> bulk(std::size_t n)
{
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<double>(i);
  }
  return v;
}

Distribution summarise_all(const std::vector<double>& v, std::size_t bins = 64)
{
  return summarise<double>(v, {}, bins);
}

}  // namespace

TEST_CASE("An all-tied array reports degeneracy, not a spread",
          "[study][distribution]")
{
  // (a). The case that matters most for this workload: lb_r is constant
  // across every subregion whose differing slots feed variables the objective
  // ignores, so "they are all the same" is an expected outcome and must be
  // reported as one rather than as a zero-width spread with infinite outlier
  // scores.
  std::vector<double> const v(1000, 7.5);
  Distribution const d = summarise_all(v);

  REQUIRE(d.count == 1000);
  CHECK(feq(d.distinct_frac, 1.0 / 1000.0));
  CHECK(feq(d.modal_frac, 1.0));
  CHECK(feq(d.iqr, 0.0));

  // The zero-dispersion gate: empty, not inf, not NaN.
  CHECK_FALSE(d.low_score.has_value());
  CHECK_FALSE(d.high_score.has_value());
  for (std::size_t i = 0; i < d.low_gap.size(); ++i) {
    INFO("isolation rank " << kIsolationRanks[i]);
    CHECK_FALSE(d.low_gap[i].has_value());
    CHECK_FALSE(d.high_gap[i].has_value());
  }
  CHECK(feq(d.low_frac, 0.0));
  CHECK(feq(d.high_frac, 0.0));

  // Every quantile is the tied value, and the mean is defined.
  for (double q : d.quantiles) {
    CHECK(feq(q, 7.5));
  }
  REQUIRE(d.mean.has_value());
  CHECK(feq(*d.mean, 7.5));
  REQUIRE(d.sd.has_value());
  CHECK(feq(*d.sd, 0.0));
}

TEST_CASE("A uniform bulk shows no outliers", "[study][distribution]")
{
  // The control. If this reported outliers, the planted cases below would
  // prove nothing.
  Distribution const d = summarise_all(bulk(1000));

  REQUIRE(d.count == 1000);
  CHECK(feq(d.distinct_frac, 1.0));
  CHECK(feq(d.modal_frac, 1.0 / 1000.0));
  CHECK(d.iqr > 0.0);

  REQUIRE(d.low_score.has_value());
  REQUIRE(d.high_score.has_value());
  // For a uniform sample both tails sit about half an IQR outside the
  // quartiles -- nowhere near Tukey's 1.5 fence.
  CHECK(*d.low_score < 1.0);
  CHECK(*d.high_score < 1.0);
  CHECK(feq(d.low_frac, 0.0));
  CHECK(feq(d.high_frac, 0.0));
}

TEST_CASE("A single planted outlier is distinguishable from a cluster",
          "[study][distribution]")
{
  // (b) and (c), the pair that layer 4 exists to separate. Both have a
  // negligible `low_frac`; only the isolation gaps tell them apart, which is
  // exactly the situation at N = 2^20 where a lone minimum and a cluster of
  // fifty both round to 0.00.
  Distribution single;
  Distribution cluster;

  {
    std::vector<double> v = bulk(1000);
    v.push_back(-10000.0);  // one, far below
    single = summarise_all(v);
  }
  {
    std::vector<double> v = bulk(1000);
    for (int i = 0; i < 10; ++i) {
      v.push_back(-10000.0 + i);  // ten, all far below
    }
    cluster = summarise_all(v);
  }

  REQUIRE(single.low_score.has_value());
  REQUIRE(cluster.low_score.has_value());

  SECTION("both are flagged as having a heavy low tail")
  {
    CHECK(*single.low_score > 10.0);
    CHECK(*cluster.low_score > 10.0);
    // ... and in both cases the mass involved is negligible, which is why a
    // fraction alone cannot distinguish them.
    CHECK(single.low_frac < 0.01);
    CHECK(cluster.low_frac < 0.02);
  }

  SECTION("only the isolation gaps separate one from ten")
  {
    // kIsolationRanks = {2, 10, 100}.
    REQUIRE(single.low_gap[0].has_value());
    REQUIRE(cluster.low_gap[0].has_value());

    // The lone outlier stands alone: the jump from the minimum to the second
    // smallest is itself many IQRs.
    CHECK(*single.low_gap[0] > 10.0);

    // The cluster does not: its second smallest is a neighbour.
    CHECK(*cluster.low_gap[0] < 0.1);

    // But by rank 100 the cluster has been left behind and the bulk reached,
    // so the gap is large again -- the signature of "a small cluster sets the
    // floor" as opposed to "one subregion does".
    REQUIRE(cluster.low_gap[2].has_value());
    CHECK(*cluster.low_gap[2] > 10.0);
  }

  SECTION("the high tail is unaffected by a planted low outlier")
  {
    // Layer 3 must be per-tail; a symmetric dispersion measure would smear
    // the planted low tail into the high one.
    REQUIRE(single.high_score.has_value());
    CHECK(*single.high_score < 1.0);
    CHECK(feq(single.high_frac, 0.0));
  }
}

TEST_CASE("Quantiles match hand-computed order statistics",
          "[study][distribution]")
{
  // (d). Linear interpolation between order statistics -- numpy/pandas'
  // default -- so that a reader can reproduce these in the tool they will
  // plot them in.
  std::vector<double> const v {1.0, 2.0, 3.0, 4.0};
  Distribution const d = summarise_all(v, 0);

  REQUIRE(d.quantiles.size() == kQuantileCount);
  CHECK(feq(d.quantiles.front(), 1.0));  // q(0)   = min
  CHECK(feq(d.quantiles.back(), 4.0));  // q(1)   = max

  // q(0.25) = 1.75, q(0.5) = 2.5, q(0.75) = 3.25 on {1,2,3,4}.
  CHECK(feq(d.iqr, 1.5));

  // With only four values the 99.9th percentile cannot be distinct from the
  // max by much, but it must still be strictly below it and must not clamp.
  double const q999 = d.quantiles[kQuantileCount - 2];
  CHECK(q999 < 4.0);
  CHECK(q999 > 3.99);

  CHECK(feq(d.distinct_frac, 1.0));
  REQUIRE(d.mean.has_value());
  CHECK(feq(*d.mean, 2.5));
}

TEST_CASE("Degenerate inputs are reported rather than divided by",
          "[study][distribution]")
{
  // (e). Each of these is a real outcome of a launch, not a malformed input.

  SECTION("an all-excluded launch summarises to nothing")
  {
    std::vector<double> const v = bulk(100);
    std::vector<unsigned char> const none(100, 0);
    Distribution const d = summarise<double>(v, none);
    CHECK(d.empty());
    CHECK(d.count == 0);
    CHECK(d.quantiles.empty());
    CHECK_FALSE(d.low_score.has_value());
  }

  SECTION("a mask selects only the unexcluded entries")
  {
    std::vector<double> const v {5.0, 1000.0, 6.0, 2000.0, 7.0};
    std::vector<unsigned char> const mask {1, 0, 1, 0, 1};
    Distribution const d = summarise<double>(v, mask);
    REQUIRE(d.count == 3);
    CHECK(feq(d.quantiles.front(), 5.0));
    CHECK(feq(d.quantiles.back(), 7.0));  // the 2000s are excluded, not clamped
  }

  SECTION("a single value is summarised without dividing by a zero span")
  {
    std::vector<double> const v {42.0};
    Distribution const d = summarise_all(v);
    REQUIRE(d.count == 1);
    CHECK(feq(d.distinct_frac, 1.0));
    CHECK(feq(d.modal_frac, 1.0));
    CHECK(feq(d.iqr, 0.0));
    CHECK_FALSE(d.low_score.has_value());
    for (double q : d.quantiles) {
      CHECK(feq(q, 42.0));
    }
  }

  SECTION("an unbounded relaxation yields no mean and no outlier scores")
  {
    // An infinite IQR makes every distance zero and every conclusion drawn
    // from it spurious, so layers 3 and 4 must decline rather than report.
    std::vector<double> v = bulk(100);
    v.push_back(-std::numeric_limits<double>::infinity());
    Distribution const d = summarise_all(v);

    REQUIRE(d.count == 101);
    CHECK(d.nonfinite_frac > 0.0);
    CHECK_FALSE(d.mean.has_value());
    CHECK_FALSE(d.sd.has_value());
    CHECK(std::isinf(d.quantiles.front()));
  }

  SECTION("an all-infinite array does not produce a NaN IQR score")
  {
    std::vector<double> const v(50,
                                std::numeric_limits<double>::infinity());
    Distribution const d = summarise_all(v);
    REQUIRE(d.count == 50);
    CHECK(feq(d.nonfinite_frac, 1.0));
    CHECK_FALSE(d.low_score.has_value());
    CHECK_FALSE(d.mean.has_value());
  }
}

TEST_CASE("The histogram accounts for every summarised value",
          "[study][distribution]")
{
  std::vector<double> const v = bulk(1000);
  Distribution const d = summarise_all(v, 32);

  REQUIRE(d.histogram.size() == 32);
  std::size_t total = 0;
  for (std::size_t c : d.histogram) {
    total += c;
  }
  CHECK(total == d.count);  // the max must land in the last bin, not past it

  SECTION("bins can be suppressed")
  {
    CHECK(summarise_all(v, 0).histogram.empty());
  }

  SECTION("a tied array puts everything in one bin")
  {
    std::vector<double> const tied(100, 3.0);
    Distribution const t = summarise_all(tied, 16);
    REQUIRE(t.histogram.size() == 16);
    CHECK(t.histogram[0] == 100);
  }
}

TEST_CASE("Concentration measures how much sits near the hull floor",
          "[study][distribution]")
{
  // Layer 5, the tie between Q2 and Q1: a tiny c(0.01) says the hull's floor
  // is set by a handful of subregions and the rest are prunable.

  SECTION("a lone minimum gives a negligible concentration")
  {
    std::vector<double> v(1000, 100.0);
    v[0] = 0.0;  // one subregion sets the floor
    double const c = concentration<double>(v, {}, 0.0, 100.0, 0.01);
    CHECK(feq(c, 1.0 / 1000.0));
  }

  SECTION("an all-tied array puts everything at the floor")
  {
    // Every subregion looks equally good to the relaxation -- the case where
    // a tight width ratio overstates the search benefit.
    std::vector<double> const v(500, 0.0);
    CHECK(feq(concentration<double>(v, {}, 0.0, 10.0, 0.01), 1.0));
  }

  SECTION("eps widens the band monotonically")
  {
    std::vector<double> const v = bulk(1000);  // 0 .. 999
    double const c01 = concentration<double>(v, {}, 0.0, 1000.0, 0.01);
    double const c05 = concentration<double>(v, {}, 0.0, 1000.0, 0.05);
    double const c25 = concentration<double>(v, {}, 0.0, 1000.0, 0.25);
    CHECK(c01 < c05);
    CHECK(c05 < c25);
    // A uniform bulk should put about eps of its mass within eps of the floor.
    CHECK(std::abs(c25 - 0.25) < 0.01);
  }

  SECTION("a degenerate hull span yields zero rather than a division")
  {
    std::vector<double> const v(10, 1.0);
    CHECK(feq(concentration<double>(v, {}, 1.0, 1.0, 0.01), 0.0));
  }

  SECTION("the mask is respected")
  {
    std::vector<double> const v {0.0, 0.0, 100.0, 100.0};
    std::vector<unsigned char> const mask {0, 1, 1, 1};
    // Of the three unexcluded, one is at the floor.
    CHECK(feq(concentration<double>(v, mask, 0.0, 100.0, 0.01), 1.0 / 3.0));
  }
}
