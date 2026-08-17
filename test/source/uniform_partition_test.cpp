// design/REFINEMENT_STUDY.md §7.2: UniformCompositionPolicy is plain
// templated C++ over cu::interval<T>/VarKind, no CUDA type in reach -- same
// reasoning as composition_policy_test.cpp for the policies it tests, so
// this needs no CUDA device.

#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/config/calibration.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/study/uniform_partition.hpp"

using cuminlp::InvalidConfiguration;
using cuminlp::ResourceExhausted;
using cuminlp::model::VarKind;
using cuminlp::region::composition_fan_out;
using cuminlp::region::SlotKind;
using cuminlp::study::UniformCompositionPolicy;

TEST_CASE("Every live dimension gets the same width, none left dead",
          "[study][uniform_partition]")
{
  std::vector<cu::interval<double>> box = {
      {0.0, 10.0}, {-5.0, 5.0}, {2.0, 2.0} /* fixed */, {0.0, 1.0}};
  std::vector<VarKind> kinds(4, VarKind::Continuous);

  UniformCompositionPolicy<double> const policy(3);  // k = 3, N_dim = 8
  auto const a = policy.choose(box, kinds);

  // The fixed dimension (index 2) never gets a slot; the three live ones
  // all do, each at width 2^3 = 8 -- Definition 6.2's "every X_i split into
  // the same N parts", not a subset chosen by width.
  REQUIRE(a.size() == 3);
  for (std::size_t j = 0; j < a.size(); ++j) {
    CHECK(a.composition[j] == SlotKind::Continuous);
    CHECK(a.widths[j] == 8);
    CHECK(a.domain_sizes[j] == 8);  // exact split, never padded
  }
  std::vector<std::size_t> const var_ids(a.var_ids.begin(), a.var_ids.end());
  CHECK(var_ids == std::vector<std::size_t> {0, 1, 3});
}

TEST_CASE("n_regions matches the actual fan-out, exponential in live count",
          "[study][uniform_partition]")
{
  std::vector<cu::interval<double>> box2 = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<VarKind> kinds2(2, VarKind::Continuous);

  for (std::size_t k : {std::size_t {1}, std::size_t {2}, std::size_t {5}}) {
    UniformCompositionPolicy<double> const policy(k);
    auto const a = policy.choose(box2, kinds2);
    std::size_t const want = std::size_t {1} << (k * 2);
    CHECK(policy.n_regions(a.composition) == want);
    CHECK(composition_fan_out(a.widths) == want);
  }
}

TEST_CASE("A live Binary or Integer variable is a hard error",
          "[study][uniform_partition]")
{
  UniformCompositionPolicy<double> const policy(2);

  SECTION("binary")
  {
    std::vector<cu::interval<double>> box = {{0.0, 10.0}, {0.0, 1.0}};
    std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Binary};
    CHECK_THROWS_AS(policy.choose(box, kinds), InvalidConfiguration);
  }

  SECTION("integer")
  {
    std::vector<cu::interval<double>> box = {{0.0, 10.0}, {0.0, 20.0}};
    std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Integer};
    CHECK_THROWS_AS(policy.choose(box, kinds), InvalidConfiguration);
  }

  SECTION("a fixed discrete variable is not live and does not throw")
  {
    std::vector<cu::interval<double>> box = {{0.0, 10.0}, {3.0, 3.0}};
    std::vector<VarKind> kinds = {VarKind::Continuous, VarKind::Integer};
    auto const a = policy.choose(box, kinds);
    REQUIRE(a.size() == 1);
    CHECK(a.var_ids[0] == 0);
  }
}

TEST_CASE("k * n_live over 62 throws ResourceExhausted before overflowing",
          "[study][uniform_partition]")
{
  std::vector<cu::interval<double>> box(7, cu::interval<double> {0.0, 1.0});
  std::vector<VarKind> kinds(7, VarKind::Continuous);

  UniformCompositionPolicy<double> const ok(8);  // 8 * 7 = 56, fine
  CHECK_NOTHROW(ok.choose(box, kinds));

  UniformCompositionPolicy<double> const over(9);  // 9 * 7 = 63
  CHECK_THROWS_AS(over.choose(box, kinds), ResourceExhausted);
}

TEST_CASE("A fully-resolved box returns the empty assignment",
          "[study][uniform_partition]")
{
  std::vector<cu::interval<double>> box = {{1.0, 1.0}, {2.0, 2.0}};
  std::vector<VarKind> kinds(2, VarKind::Continuous);

  UniformCompositionPolicy<double> const policy(4);
  auto const a = policy.choose(box, kinds);
  CHECK(a.size() == 0);
}

TEST_CASE("k must be at least 1", "[study][uniform_partition]")
{
  CHECK_THROWS_AS(UniformCompositionPolicy<double>(0), InvalidConfiguration);
}
