// The CUDA-graph backend against the role contract every backend must pass
// (design/MODULE_REFACTOR.md §12 stage 3). The contract itself is in
// backend_contract.hpp and names no CUDA type; this file supplies a model and
// a factory and does nothing else, which is exactly what a second backend
// will do. Requires an actual CUDA device.

#include <cstddef>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_contract.hpp"
#include "cuminlp/backend/graph/factory.cuh"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"

using cuminlp::model::Expr;
using cuminlp::model::Problem;
using cuminlp::region::Composition;
using cuminlp::region::FanOutSpec;
using cuminlp::region::SlotAssignment;
using cuminlp::region::SlotKind;

namespace
{

// Two binaries, two integers on [0, 3] and two continuous on [-2, 2], with a
// separable objective the host can recompute in one line and a constraint
// that actually excludes part of the box.
Problem<double> make_model()
{
  Problem<double> p;
  std::vector<Expr<double>> v;
  v.push_back(p.bin_var());
  v.push_back(p.bin_var());
  v.push_back(p.int_var(0.0, 3.0));
  v.push_back(p.int_var(0.0, 3.0));
  v.push_back(p.var(-2.0, 2.0));
  v.push_back(p.var(-2.0, 2.0));

  Expr<double> obj = (v[0] - 1.0) * (v[0] - 1.0);
  for (std::size_t i = 1; i < v.size(); ++i) {
    obj = obj + (v[i] - 1.0) * (v[i] - 1.0);
  }
  p.set_objective(obj);
  p.add_constraint(v[0] + v[1], cuminlp::model::Cmp::LE, 1.0);
  return p;
}

double reference_objective(std::span<const double> x)
{
  double total = 0.0;
  for (double xi : x) {
    total += (xi - 1.0) * (xi - 1.0);
  }
  return total;
}

bool reference_feasible(std::span<const double> x)
{
  return x[0] + x[1] <= 1.0 + 1e-9;
}

}  // namespace

TEST_CASE("The CUDA-graph backend satisfies the region-role contract",
          "[backend][contract]")
{
  Problem<double> const problem = make_model();

  cuminlp::test::ContractModel const model {
      problem,
      problem.box_bounds,
      reference_objective,
      reference_feasible,
  };

  // Both integer domains are 4 wide, so enumerate_cap = 4 enumerates them
  // rather than bisecting: the enumerable assignment below really is one.
  FanOutSpec const fan_out {2, 4};

  SlotAssignment const enumerable {
      Composition {.kinds = {SlotKind::BinaryEnumerate,
                             SlotKind::BinaryEnumerate,
                             SlotKind::IntegerEnumerate,
                             SlotKind::IntegerEnumerate}},
      {0, 1, 2, 3},
      {2, 2, 4, 4},
  };
  SlotAssignment const subdividing {
      Composition {.kinds = {SlotKind::Continuous, SlotKind::Continuous}},
      {4, 5},
      {2, 2},
  };

  cuminlp::backend::graph::GraphBackendFactory<double> const factory;
  cuminlp::test::check_backend_contract(factory,
                                        model,
                                        enumerable,
                                        subdividing,
                                        fan_out,
                                        /*samples_per_region=*/4);
}
