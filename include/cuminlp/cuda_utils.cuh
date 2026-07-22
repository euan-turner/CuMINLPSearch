#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cuminlp::detail
{

constexpr std::size_t ceil_div(std::size_t x, std::size_t y) { return (x + y - 1) / y; }

inline void check(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
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
                                Func func, dim3 grid, dim3 block,
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
  check(cudaGraphAddKernelNode(&node, graph, deps.empty() ? nullptr : deps.data(), deps.size(), &params),
       "cudaGraphAddKernelNode");
  return node;
}

}  // namespace cuminlp::detail
