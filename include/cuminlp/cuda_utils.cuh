#pragma once

#include <cstddef>
#include <cstdio>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuminlp/errors.hpp"
#include "cuminlp/saturating_arith.hpp"

namespace cuminlp
{

// A CUDA runtime/driver call failed. Carries the cudaError_t so callers
// can branch on the specific failure, not just the message.
class CUDAError : public error
{
public:
  CUDAError(cudaError_t code, const char* api_call)
      : error(std::string(api_call) + " failed: " + cudaGetErrorString(code))
      , code_(code)
  {
  }

  cudaError_t code() const noexcept { return code_; }

private:
  cudaError_t code_;
};

}  // namespace cuminlp

namespace cuminlp::detail
{

// ceil_div/saturating_mul/saturating_add now live in saturating_arith.hpp
// (included above), so host-only code can reuse them without this header's
// <cuda_runtime.h> dependency.

// Byte counts in diagnostics, e.g. "221.2 GiB". Plain bytes below 1 KiB.
inline std::string format_bytes(std::size_t bytes)
{
  if (bytes == std::numeric_limits<std::size_t>::max()) {
    return "more than 2^64 bytes (the size computation saturated)";
  }
  static const char* const units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
  auto value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(units)) {
    value /= 1024.0;
    ++unit;
  }
  char buf[64];
  std::snprintf(
      buf, sizeof(buf), unit == 0 ? "%.0f %s" : "%.1f %s", value, units[unit]);
  return buf;
}

// Thousands separators, so a nine-digit region count is readable at a glance.
inline std::string format_count(std::size_t n)
{
  std::string s = std::to_string(n);
  for (std::size_t pos = s.size() > 3 ? s.size() - 3 : 0; pos > 0; pos -= 3) {
    s.insert(pos, ",");
    if (pos < 3) {
      break;
    }
  }
  return s;
}

template<std::size_t Base, std::size_t Exp>
constexpr std::size_t pown()
{
  if constexpr (Exp == 0) {
    return 1;
  } else {
    return Base * pown<Base, Exp - 1>();
  }
}

inline void check(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw cuminlp::CUDAError(err, what);
  }
}

template<typename U>
U* alloc_device(std::size_t n)
{
  U* p = nullptr;
  check(cudaMalloc(&p, n * sizeof(U)), "cudaMalloc");
  return p;
}

// Builds one cudaKernelNodeParams and adds it as a node in `graph` in a single
// step, so the argument copies backing `kernelParams` stay alive for the
// duration of the call (the driver copies their bytes before returning).
template<typename Func, typename... Args>
cudaGraphNode_t add_kernel_node(cudaGraph_t graph,
                                const std::vector<cudaGraphNode_t>& deps,
                                Func func,
                                dim3 grid,
                                dim3 block,
                                Args... args)
{
  void* kernel_args[] = {const_cast<void*>(static_cast<const void*>(&args))...};
  cudaKernelNodeParams params {};
  params.func = reinterpret_cast<void*>(func);
  params.gridDim = grid;
  params.blockDim = block;
  params.sharedMemBytes = 0;
  params.kernelParams = kernel_args;
  params.extra = nullptr;

  cudaGraphNode_t node;
  check(cudaGraphAddKernelNode(&node,
                               graph,
                               deps.empty() ? nullptr : deps.data(),
                               deps.size(),
                               &params),
        "cudaGraphAddKernelNode");
  return node;
}

}  // namespace cuminlp::detail
