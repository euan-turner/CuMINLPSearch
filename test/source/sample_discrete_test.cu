#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/cuda_utils.cuh"
#include "cuminlp/dag.hpp"
#include "cuminlp/graph_replay.cuh"

using cuminlp::SlotKind;
using cuminlp::dag::VarKind;

namespace
{

// Directly launches sample_points_kernel (graph_replay.cuh) -- bypassing
// GraphBuilder/GraphReplay entirely -- against a 3-variable synthetic domain:
// var 0 continuous [0,10], var 1 integer [3,9], var 2 binary [0,1]. Only var
// 0 is cycled (bisected into 4 slices); vars 1 and 2 keep their parent
// bounds unchanged, which is exactly the case that mattered here: even a
// variable this iteration isn't touching must still be sampled on the
// integer lattice if its kind isn't Continuous.
template<typename T>
struct SampleResult
{
  std::vector<T> continuous;  // var 0, n_regions * SamplePoints entries
  std::vector<T> integer;  // var 1
  std::vector<T> binary;  // var 2
};

template<typename T, std::size_t SamplePoints>
SampleResult<T> run_sample_points_kernel()
{
  constexpr std::size_t CYCLE_SIZE = 1;
  constexpr std::size_t N_VARS = 3;
  constexpr std::size_t N_REGIONS = 4;  // matches slot_fan_out[0] below

  std::vector<cu::interval<T>> domain_host = {
      {T(0), T(10)},  // var 0: continuous
      {T(3), T(9)},  // var 1: integer
      {T(0), T(1)},  // var 2: binary
  };
  std::vector<VarKind> var_kinds_host = {
      VarKind::Continuous, VarKind::Integer, VarKind::Binary};
  std::array<std::size_t, CYCLE_SIZE> slot_var_ids_host = {0};
  std::array<std::uint32_t, CYCLE_SIZE> slot_fan_out_host = {4};
  std::array<SlotKind, CYCLE_SIZE> slot_kind_host = {SlotKind::Continuous};

  auto* domain = cuminlp::detail::alloc_device<cu::interval<T>>(N_VARS);
  auto* var_kinds = cuminlp::detail::alloc_device<VarKind>(N_VARS);
  auto* slot_var_ids = cuminlp::detail::alloc_device<std::size_t>(CYCLE_SIZE);
  auto* slot_fan_out = cuminlp::detail::alloc_device<std::uint32_t>(CYCLE_SIZE);
  auto* slot_kind = cuminlp::detail::alloc_device<SlotKind>(CYCLE_SIZE);

  cuminlp::detail::check(cudaMemcpy(domain,
                                    domain_host.data(),
                                    N_VARS * sizeof(cu::interval<T>),
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy");
  cuminlp::detail::check(cudaMemcpy(var_kinds,
                                    var_kinds_host.data(),
                                    N_VARS * sizeof(VarKind),
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy");
  cuminlp::detail::check(cudaMemcpy(slot_var_ids,
                                    slot_var_ids_host.data(),
                                    CYCLE_SIZE * sizeof(std::size_t),
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy");
  cuminlp::detail::check(cudaMemcpy(slot_fan_out,
                                    slot_fan_out_host.data(),
                                    CYCLE_SIZE * sizeof(std::uint32_t),
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy");
  cuminlp::detail::check(cudaMemcpy(slot_kind,
                                    slot_kind_host.data(),
                                    CYCLE_SIZE * sizeof(SlotKind),
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy");

  constexpr std::size_t N_ELEMS = N_REGIONS * SamplePoints;
  std::array<T*, N_VARS> var_buffer_host {};
  for (auto& buf : var_buffer_host) {
    buf = cuminlp::detail::alloc_device<T>(N_ELEMS);
  }
  auto* var_buffers = cuminlp::detail::alloc_device<T*>(N_VARS);
  cuminlp::detail::check(cudaMemcpy(var_buffers,
                                    var_buffer_host.data(),
                                    N_VARS * sizeof(T*),
                                    cudaMemcpyHostToDevice),
                         "cudaMemcpy");

  cuminlp::dag::sample_points_kernel<T, CYCLE_SIZE>
      <<<1, 256>>>(domain,
                   slot_var_ids,
                   slot_fan_out,
                   slot_kind,
                   var_kinds,
                   var_buffers,
                   N_VARS,
                   N_REGIONS,
                   /*slot_count=*/CYCLE_SIZE,
                   SamplePoints,
                   /*salt=*/42);
  cuminlp::detail::check(cudaGetLastError(), "sample_points_kernel launch");
  cuminlp::detail::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  SampleResult<T> result;
  result.continuous.resize(N_ELEMS);
  result.integer.resize(N_ELEMS);
  result.binary.resize(N_ELEMS);
  cuminlp::detail::check(cudaMemcpy(result.continuous.data(),
                                    var_buffer_host[0],
                                    N_ELEMS * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy");
  cuminlp::detail::check(cudaMemcpy(result.integer.data(),
                                    var_buffer_host[1],
                                    N_ELEMS * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy");
  cuminlp::detail::check(cudaMemcpy(result.binary.data(),
                                    var_buffer_host[2],
                                    N_ELEMS * sizeof(T),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy");

  cudaFree(domain);
  cudaFree(var_kinds);
  cudaFree(slot_var_ids);
  cudaFree(slot_fan_out);
  cudaFree(slot_kind);
  for (auto* buf : var_buffer_host) {
    cudaFree(buf);
  }
  cudaFree(var_buffers);

  return result;
}

}  // namespace

TEMPLATE_TEST_CASE(
    "sample_points_kernel snaps Integer/Binary variables to the integer "
    "lattice",
    "[sample_points_kernel]",
    float,
    double)
{
  using T = TestType;
  constexpr std::size_t SAMPLE_POINTS = 50;

  auto result = run_sample_points_kernel<T, SAMPLE_POINTS>();

  std::set<T> distinct_integer_values;
  for (T v : result.integer) {
    CHECK(v == std::floor(v));  // exact integer, not just close to one
    CHECK(v >= T(3));
    CHECK(v <= T(9));
    distinct_integer_values.insert(v);
  }
  // Sanity check the sampler isn't degenerately returning the same value
  // every time -- weak, but would catch a broken RNG/indexing bug.
  CHECK(distinct_integer_values.size() > 1);

  for (T v : result.binary) {
    CHECK((v == T(0) || v == T(1)));
  }

  for (T v : result.continuous) {
    CHECK(v >= T(0));
    CHECK(v <= T(10));
  }
}
