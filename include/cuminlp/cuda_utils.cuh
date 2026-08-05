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
#include "cuminlp/format.hpp"
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

// ceil_div/saturating_mul/saturating_add now live in saturating_arith.hpp,
// and format_bytes/format_count in format.hpp (both included above), so
// host-only code can reuse them without this header's <cuda_runtime.h>
// dependency.

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
