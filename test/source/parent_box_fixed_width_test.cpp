// design/REFINEMENT_STUDY.md §7.3: study::draw_fixed_width_box is plain
// templated C++ over cu::interval<T>/VarKind, same as draw_parent_box in the
// same header -- no CUDA device needed.

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/model/problem.hpp"
#include "cuminlp/study/parent_box.hpp"

using cuminlp::model::VarKind;
using cuminlp::study::draw_fixed_width_box;
using cuminlp::study::draw_parent_box;
using cuminlp::study::inside;
using Box = std::vector<cu::interval<double>>;

namespace
{

/// Exact equality without tripping -Werror=float-equal, the same idiom
/// study_distribution_test.cpp uses. Every value checked below is either a
/// planted root/width constant or its exact clamp/snap, so anything but
/// exact agreement is a real disagreement, not rounding.
bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

}  // namespace

TEST_CASE("Every live variable gets width min(W, root width)",
          "[study][parent_box]")
{
  Box const root = {{-1000000.0, 1000000.0}, {1.0, 2.0}, {5.0, 5.0}};
  std::vector<VarKind> const kinds(3, VarKind::Continuous);

  std::mt19937_64 rng(20260814);
  Box const box = draw_fixed_width_box<double>(root, kinds, 1.0, rng);

  REQUIRE(box.size() == 3);
  // Variable 0: root width 2e6, requested width 1 -- achieves ~1, up to the
  // usual (centre +/- half) rounding (not bit-exact for an arbitrary drawn
  // centre).
  CHECK(std::abs((box[0].ub - box[0].lb) - 1.0) < 1e-9);
  // Variable 1: root width 1, requested width 1 -- clamps to the root's own
  // width. This window is degenerate (a single point), so the result is
  // deterministic and exact.
  CHECK(feq(box[1].ub - box[1].lb, 1.0));
  CHECK(inside<double>(box, root));
}

TEST_CASE("A requested width wider than the root clamps to the root",
          "[study][parent_box]")
{
  Box const root = {{0.0, 0.5}};
  std::vector<VarKind> const kinds = {VarKind::Continuous};

  std::mt19937_64 rng(1);
  Box const box = draw_fixed_width_box<double>(root, kinds, 1.0, rng);
  CHECK(feq(box[0].lb, 0.0));
  CHECK(feq(box[0].ub, 0.5));
}

TEST_CASE("A variable fixed in the root stays fixed",
          "[study][parent_box]")
{
  Box const root = {{3.0, 3.0}, {-10.0, 10.0}};
  std::vector<VarKind> const kinds(2, VarKind::Continuous);

  std::mt19937_64 rng(7);
  Box const box = draw_fixed_width_box<double>(root, kinds, 1.0, rng);
  CHECK(feq(box[0].lb, 3.0));
  CHECK(feq(box[0].ub, 3.0));
  CHECK(std::abs((box[1].ub - box[1].lb) - 1.0) < 1e-9);
}

TEST_CASE("The same seed reproduces the same boxes",
          "[study][parent_box]")
{
  Box const root = {{-100.0, 100.0}, {-50.0, 50.0}};
  std::vector<VarKind> const kinds(2, VarKind::Continuous);

  std::mt19937_64 a(42), b(42);
  Box const box_a = draw_fixed_width_box<double>(root, kinds, 1.0, a);
  Box const box_b = draw_fixed_width_box<double>(root, kinds, 1.0, b);
  for (std::size_t v = 0; v < box_a.size(); ++v) {
    CHECK(feq(box_a[v].lb, box_b[v].lb));
    CHECK(feq(box_a[v].ub, box_b[v].ub));
  }
}

TEST_CASE("Every draw is contained in the root over many placements",
          "[study][parent_box]")
{
  Box const root = {{-1000000.0, 1000000.0}, {-3.0, 3.0}};
  std::vector<VarKind> const kinds(2, VarKind::Continuous);

  std::mt19937_64 rng(99);
  for (int rep = 0; rep < 200; ++rep) {
    Box const box = draw_fixed_width_box<double>(root, kinds, 1.0, rng);
    INFO("rep " << rep);
    REQUIRE(inside<double>(box, root));
  }
}

TEST_CASE("center_range keeps placements near the origin despite a huge root",
          "[study][parent_box]")
{
  // Mirrors ex4_1_5's actual root: one side declared, the other defaulted
  // to a synthetic +-1,000,000 (design/REFINEMENT_STUDY.md §7.2's finding).
  // Without center_range, a width-1 draw over this root lands out in the
  // padding with near certainty; center_range=10 should keep every draw
  // near 0 instead, wherever the origin is reachable while staying in root.
  Box const root = {{-5.0, 1000000.0}, {-1000000.0, 5.0}};
  std::vector<VarKind> const kinds(2, VarKind::Continuous);

  std::mt19937_64 rng(20260814);
  for (int rep = 0; rep < 100; ++rep) {
    Box const box =
        draw_fixed_width_box<double>(root, kinds, 1.0, rng, 10.0);
    INFO("rep " << rep << " box0 [" << box[0].lb << ", " << box[0].ub
                << "] box1 [" << box[1].lb << ", " << box[1].ub << "]");
    REQUIRE(inside<double>(box, root));
    // The achieved width is still ~1 (root is wide enough everywhere near
    // 0), up to the usual (centre +/- half) rounding -- not exact equality,
    // since an arbitrary drawn centre is not guaranteed to be a double for
    // which +/- half round-trips bit-exactly.
    CHECK(std::abs((box[0].ub - box[0].lb) - 1.0) < 1e-9);
    CHECK(std::abs((box[1].ub - box[1].lb) - 1.0) < 1e-9);
    CHECK(box[0].lb >= -10.5);
    CHECK(box[0].ub <= 10.5);
    CHECK(box[1].lb >= -10.5);
    CHECK(box[1].ub <= 10.5);
  }
}

TEST_CASE("Discrete snapping matches draw_parent_box's outward-then-clamp "
          "behaviour",
          "[study][parent_box]")
{
  Box const root = {{-20.0, 20.0}};
  std::vector<VarKind> const kinds = {VarKind::Integer};

  std::mt19937_64 rng(4242);
  for (int rep = 0; rep < 50; ++rep) {
    Box const box = draw_fixed_width_box<double>(root, kinds, 1.0, rng);
    double const lo = box[0].lb;
    double const hi = box[0].ub;
    INFO("rep " << rep << " box [" << lo << ", " << hi << "]");
    REQUIRE(feq(lo, std::floor(lo)));
    REQUIRE(feq(hi, std::floor(hi)));
    REQUIRE(hi >= lo);
    REQUIRE(inside<double>(box, root));
  }
}
