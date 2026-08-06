#pragma once

#include <cmath>
#include <type_traits>

#include <cuinterval/cuinterval.h>
#include <cuinterval/interval.h>

#include "cuminlp/model/dag.hpp"

// Device functor tags for each DAGNode::Op, and the op_code<> trait mapping
// each tag back to the Op it implements -- moved out of graph_replay.cuh
// (design/MODULE_REFACTOR.md §11).
namespace cuminlp::backend::graph
{

using cuminlp::model::Op;

// Functor wrappers for Op

// Each tag overloads apply() for interval-interval, interval-scalar, and
// scalar-interval operands. min/max lack a mixed overload in cuinterval, so
// their scalar apply() wraps the scalar as a point interval.
struct AddOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a + b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a + b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a + b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a + b;
  }
};

struct SubOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a - b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a - b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a - b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a - b;
  }
};

struct MulOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a * b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a * b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a * b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a * b;
  }
};

struct DivOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return a / b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return a / b;
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return a / b;
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return a / b;
  }
};

struct MinOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return min(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return min(a, cu::interval<T>(b));
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return min(cu::interval<T>(a), b);
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return cu::intrinsic::min<T>(a, b);
  }
};

struct MaxOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return max(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return max(a, cu::interval<T>(b));
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return max(cu::interval<T>(a), b);
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    return cu::intrinsic::max<T>(a, b);
  }
};

// a^b with both a and b general (interval or scalar), as opposed to
// pown_kernel's integer-exponent path below. cu::pow(interval, interval)
// clips a negative-base domain to [0, +inf) internally, same as cu::sqrt/
// cu::log do for their domains, so no extra domain handling is needed here.
struct PowOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, cu::interval<T> b)
  {
    return pow(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a, T b)
  {
    return pow(a, b);
  }

  template<typename T>
  static __device__ cu::interval<T> apply(T a, cu::interval<T> b)
  {
    return pow(cu::interval<T>(a), b);
  }

  template<typename T>
  static __device__ T apply(T a, T b)
  {
    if constexpr (std::is_same_v<T, float>) {
      return ::powf(a, b);
    } else {
      using std::pow;
      return pow(a, b);
    }
  }
};

struct NegOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return -a;
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return -a;
  }
};

struct SqrOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return sqr(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return a * a;
  }
};

struct ExpOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return exp(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::exp(a);
  }
};

struct LogOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return log(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::log(a);
  }
};

struct SqrtOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return sqrt(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::sqrt(a);
  }
};

struct SinOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return sin(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::sin(a);
  }
};

struct CosOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return cos(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::cos(a);
  }
};

struct TanhOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return tanh(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::tanh(a);
  }
};

struct AbsOp
{
  template<typename T>
  static __device__ cu::interval<T> apply(cu::interval<T> a)
  {
    return abs(a);
  }

  template<typename T>
  static __device__ T apply(T a)
  {
    return std::fabs(a);
  }
};

// Maps each Op functor tag back to the DAGNode::Op it implements, so
// wire_binary/wire_unary can assert the node they were handed actually
// matches the template parameter the ensure_node switch dispatched to.
template<class OpTag>
struct op_code;

template<>
struct op_code<AddOp>
{
  static constexpr Op value = Op::Add;
};

template<>
struct op_code<SubOp>
{
  static constexpr Op value = Op::Sub;
};

template<>
struct op_code<MulOp>
{
  static constexpr Op value = Op::Mul;
};

template<>
struct op_code<DivOp>
{
  static constexpr Op value = Op::Div;
};

template<>
struct op_code<MinOp>
{
  static constexpr Op value = Op::Min;
};

template<>
struct op_code<MaxOp>
{
  static constexpr Op value = Op::Max;
};

template<>
struct op_code<PowOp>
{
  static constexpr Op value = Op::Pow;
};

template<>
struct op_code<NegOp>
{
  static constexpr Op value = Op::Neg;
};

template<>
struct op_code<SqrOp>
{
  static constexpr Op value = Op::Sqr;
};

template<>
struct op_code<ExpOp>
{
  static constexpr Op value = Op::Exp;
};

template<>
struct op_code<LogOp>
{
  static constexpr Op value = Op::Log;
};

template<>
struct op_code<SqrtOp>
{
  static constexpr Op value = Op::Sqrt;
};

template<>
struct op_code<SinOp>
{
  static constexpr Op value = Op::Sin;
};

template<>
struct op_code<CosOp>
{
  static constexpr Op value = Op::Cos;
};

template<>
struct op_code<TanhOp>
{
  static constexpr Op value = Op::Tanh;
};

template<>
struct op_code<AbsOp>
{
  static constexpr Op value = Op::Abs;
};

template<typename T>
__device__ T pown_scalar(T base, int exponent)
{
  if constexpr (std::is_same_v<T, float>) {
    return ::powf(base, exponent);
  } else {
    using std::pow;
    return pow(base, exponent);
  }
}

}  // namespace cuminlp::backend::graph
