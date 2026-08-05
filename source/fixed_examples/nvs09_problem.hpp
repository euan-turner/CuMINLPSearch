#pragma once

/**
 * @brief The nvs09 MINLPLib instance (https://www.minlplib.org/nvs09.html) as
 * a hand-built `dag::Problem`.
 */

#include <cstddef>
#include <vector>

#include "cuminlp/dag.hpp"

namespace cuminlp::examples::nvs09
{

constexpr std::size_t NUM_VARS = 10;
// 7, not NUM_VARS. Every variable is an integer on [3, 9], i.e. domain size
// 7 <= enumerate_cap, so all of them enumerate rather than bisect and the
// region count is 7^CYCLE_SIZE. At CYCLE_SIZE = 10 that is 7^10 = 282M
// regions, which at 841 B each (102 DAG-node buffers plus bookkeeping) is
// 221 GiB -- this instance has never actually run on a consumer GPU, it just
// failed with a bare cudaMalloc OOM. 7^7 = 823543 regions is 0.65 GiB and
// solves to the published optimum (-43.134336).
constexpr std::size_t CYCLE_SIZE = 7;
constexpr std::size_t PARTITION_NUM = 7;  // every integer variable is [3, 9]
constexpr std::size_t SAMPLE_POINTS = 5;

/// @brief Build nvs09: 10 integers in [3, 9], minimising a log/product
/// objective. See the module comment above for the instance's provenance.
inline auto make_nvs09() -> dag::Problem<double>
{
  using dag::Expr;
  dag::Problem<double> problem;
  std::vector<Expr<double>> i;
  i.reserve(NUM_VARS);
  for (std::size_t j = 0; j < NUM_VARS; ++j) {
    i.push_back(problem.int_var(3.0, 9.0));
  }

  Expr<double> product = i[0];
  for (std::size_t j = 1; j < NUM_VARS; ++j) {
    product = product * i[j];
  }

  Expr<double> sum_terms = sqr(log(i[0] - 2.0)) + sqr(log(10.0 - i[0]));
  for (std::size_t j = 1; j < NUM_VARS; ++j) {
    sum_terms = sum_terms + sqr(log(i[j] - 2.0)) + sqr(log(10.0 - i[j]));
  }

  problem.set_objective(sum_terms - exp(0.2 * log(product)));
  return problem;
}

}  // namespace cuminlp::examples::nvs09
