#pragma once

// Test-only Rounding policy (see cuinterval.cuh's CudaRounding): directed
// rounding via MXCSR instead of CUDA intrinsics, so composed interval ops
// can run on the host as a test oracle. x86-only. Requires -frounding-math
// on any TU that instantiates this (see test/CMakeLists.txt) -- otherwise
// the compiler may constant-fold "a + b" under an assumed round-to-nearest
// mode; the fences/volatile guard against reordering across the mode switch.

#include <xmmintrin.h>

#include <atomic>

#define CPU_DIRECTED_ROUND(mode, expr) \
  [&] { \
    unsigned int old_round_mode_ = _MM_GET_ROUNDING_MODE(); \
    _MM_SET_ROUNDING_MODE(mode); \
    std::atomic_signal_fence(std::memory_order_seq_cst); \
    volatile auto result_ = (expr); \
    std::atomic_signal_fence(std::memory_order_seq_cst); \
    _MM_SET_ROUNDING_MODE(old_round_mode_); \
    return result_; \
  }()

struct CpuRounding
{
  static inline float add_rd(float a, float b) { return CPU_DIRECTED_ROUND(_MM_ROUND_DOWN, a + b); }
  static inline double add_rd(double a, double b) { return CPU_DIRECTED_ROUND(_MM_ROUND_DOWN, a + b); }
  static inline float add_ru(float a, float b) { return CPU_DIRECTED_ROUND(_MM_ROUND_UP, a + b); }
  static inline double add_ru(double a, double b) { return CPU_DIRECTED_ROUND(_MM_ROUND_UP, a + b); }

  static inline float sub_rd(float a, float b) { return CPU_DIRECTED_ROUND(_MM_ROUND_DOWN, a - b); }
  static inline double sub_rd(double a, double b) { return CPU_DIRECTED_ROUND(_MM_ROUND_DOWN, a - b); }
  static inline float sub_ru(float a, float b) { return CPU_DIRECTED_ROUND(_MM_ROUND_UP, a - b); }
  static inline double sub_ru(double a, double b) { return CPU_DIRECTED_ROUND(_MM_ROUND_UP, a - b); }

  static inline float mul_rd(float a, float b) { return CPU_DIRECTED_ROUND(_MM_ROUND_DOWN, a * b); }
  static inline double mul_rd(double a, double b) { return CPU_DIRECTED_ROUND(_MM_ROUND_DOWN, a * b); }
  static inline float mul_ru(float a, float b) { return CPU_DIRECTED_ROUND(_MM_ROUND_UP, a * b); }
  static inline double mul_ru(double a, double b) { return CPU_DIRECTED_ROUND(_MM_ROUND_UP, a * b); }
};
