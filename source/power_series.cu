// https://www.minlplib.org/ex4_1_2.html
// lots of powers

#include <memory>
#include <string>

#include "cuminlp/capacity_ladder.hpp"
#include "cuminlp/composition_policy.hpp"
#include "cuminlp/dag.hpp"
#include "cuminlp/example_main.hpp"
#include "cuminlp/graph_driver.cuh"

using cuminlp::dag::Expr;
using cuminlp::dag::Problem;

namespace
{
constexpr std::size_t CYCLE_SIZE = 1;
// SlotContext is register-resident, so the capacity is chosen from a compiled
// ladder rather than being CYCLE_SIZE itself; CYCLE_SIZE remains the cap the
// policy honours (see include/cuminlp/capacity_ladder.hpp).
constexpr std::size_t CAPACITY = cuminlp::ladder_rung_or_zero(CYCLE_SIZE);
static_assert(CAPACITY != 0, "CYCLE_SIZE exceeds the widest compiled rung");
constexpr std::size_t PARTITION_NUM = 10000;
constexpr std::size_t SAMPLE_POINTS = 10;

// Coefficient of x1^k for k = 3..50; the x1 and x1^2 terms are written out
// separately below because ex4_1_2 states the quadratic one as sqr(x1).
constexpr double COEFFS[] = {
    1.666666666,  // ^3
    1.25,  // ^4
    1.0,  // ^5
    0.8333333,  // ^6
    0.714285714,  // ^7
    0.625,  // ^8
    0.555555555,  // ^9
    1.0,  // ^10
    -43.6363636,  // ^11
    0.41666666,  // ^12
    0.384615384,  // ^13
    0.357142857,  // ^14
    0.3333333,  // ^15
    0.3125,  // ^16
    0.294117647,  // ^17
    0.277777777,  // ^18
    0.263157894,  // ^19
    0.25,  // ^20
    0.238095238,  // ^21
    0.227272727,  // ^22
    0.217391304,  // ^23
    0.208333333,  // ^24
    0.2,  // ^25
    0.192307692,  // ^26
    0.185185185,  // ^27
    0.178571428,  // ^28
    0.344827586,  // ^29
    0.6666666,  // ^30
    -15.48387097,  // ^31
    0.15625,  // ^32
    0.1515151,  // ^33
    0.14705882,  // ^34
    0.14285712,  // ^35
    0.138888888,  // ^36
    0.135135135,  // ^37
    0.131578947,  // ^38
    0.128205128,  // ^39
    0.125,  // ^40
    0.121951219,  // ^41
    0.119047619,  // ^42
    0.116279069,  // ^43
    0.113636363,  // ^44
    0.1111111,  // ^45
    0.108695652,  // ^46
    0.106382978,  // ^47
    0.208333333,  // ^48
    0.408163265,  // ^49
    0.8,  // ^50
};
constexpr int FIRST_POWER = 3;
constexpr std::size_t NUM_COEFFS = sizeof(COEFFS) / sizeof(COEFFS[0]);
}  // namespace

auto main(int argc, char* argv[]) -> int
{
  return cuminlp::examples::guarded(
      [&]
      {
        Problem<double> problem;
        auto x = problem.var(1, 2);

        // ex4_1_2's single equation is -(P(x1)) + objvar =E= 0, i.e. objvar =
        // P(x1), so the polynomial itself is what gets minimised over x1 in [1,
        // 2].
        Expr<double> obj = (2.5 * (x * x)) - (500.0 * x);
        for (std::size_t i = 0; i < NUM_COEFFS; ++i) {
          obj = obj + (COEFFS[i] * pow(x, FIRST_POWER + static_cast<int>(i)));
        }
        problem.set_objective(obj);

        int iters = std::stoi(argv[1]);
        auto policy = std::make_shared<
            cuminlp::GreedyCompositionPolicy<double, CAPACITY>>(
            cuminlp::FanOutSpec {PARTITION_NUM},
            cuminlp::SearchCalibration {.max_cycle_size = CYCLE_SIZE});
        cuminlp::GraphDriver<double, CAPACITY> drv(
            policy, iters, 1e-9, SAMPLE_POINTS);
        drv.solve(problem);
      });
}
