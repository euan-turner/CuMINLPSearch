#pragma once

// Internal to the GAMS frontend. Not installed, not part of the public API.

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuminlp/dag.hpp"

namespace cuminlp::gams::detail
{

enum class Kind { Num, Var, Add, Sub, Mul, Div, Neg, Call };

/// Functions the frontend can lower. Names GAMS has that are absent here are
/// rejected by name at parse time (see kUnsupported in frontend.cpp).
enum class Func { Sqr, Sqrt, Exp, Log, Log10, Log2, Abs, Sin, Cos, Tanh, Min, Max, Power };

/// Exact floating-point equality without tripping -Wfloat-equal (the `dev`
/// preset builds this as an error). Used only where exact equality really is
/// what's meant -- e.g. "is this literal exactly the fold identity 0/1/-1" --
/// never as a tolerance-based approximate comparison.
constexpr bool feq(double a, double b)
{
  return !(a < b) && !(b < a);
}

struct Node {
  Kind kind = Kind::Num;
  double value = 0.0;  // Kind::Num
  int sym = -1;        // Kind::Var: index into Model::symbols
  Func func = Func::Sqr;  // Kind::Call
  std::vector<int> args;  // child node indices; every child index < this index
  int line = 0;
};

/**
 * @brief Flat, index-addressed expression tree.
 *
 * Children are always created before parents, so a child's index is always
 * smaller than its parent's. That makes bottom-up passes a forward sweep and
 * makes rearrangement (see split() in frontend.cpp) index rewrites rather
 * than pointer surgery.
 *
 * The builders below apply a small peephole. It exists so the canonical
 * objective form `-(P) + objvar =E= 0` rearranges to exactly `P` with no
 * leftover `x - 0` / `Neg(Neg(x))` nodes, and so constant subtrees collapse
 * before they ever reach the DAG.
 *
 * Deliberately NOT folded: `0 * x` -> `0` and `0 / x` -> `0`. Both are valid
 * for reals but change the result under interval arithmetic when x is
 * unbounded, and this frontend feeds an interval evaluator. Only identities
 * that are sound for intervals are applied.
 */
class Ast
{
public:
  auto operator[](int i) const -> Node const& { return nodes_[i]; }
  auto size() const -> std::size_t { return nodes_.size(); }

  auto num(double v, int line = 0) -> int
  {
    Node n;
    n.kind = Kind::Num;
    n.value = v;
    n.line = line;
    return push(n);
  }

  auto var(int sym, int line) -> int
  {
    Node n;
    n.kind = Kind::Var;
    n.sym = sym;
    n.line = line;
    return push(n);
  }

  auto call(Func f, std::vector<int> args, int line) -> int
  {
    Node n;
    n.kind = Kind::Call;
    n.func = f;
    n.args = std::move(args);
    n.line = line;
    return push(n);
  }

  auto neg(int a) -> int
  {
    if (auto v = as_num(a)) return num(-*v, nodes_[a].line);
    if (nodes_[a].kind == Kind::Neg) return nodes_[a].args[0];  // Neg(Neg(x)) -> x
    return binary_node(Kind::Neg, {a}, nodes_[a].line);
  }

  auto add(int a, int b) -> int
  {
    auto va = as_num(a), vb = as_num(b);
    if (va && vb) return num(*va + *vb, nodes_[a].line);
    if (vb && feq(*vb, 0.0)) return a;
    if (va && feq(*va, 0.0)) return b;
    return binary_node(Kind::Add, {a, b}, nodes_[a].line);
  }

  auto sub(int a, int b) -> int
  {
    auto va = as_num(a), vb = as_num(b);
    if (va && vb) return num(*va - *vb, nodes_[a].line);
    if (vb && feq(*vb, 0.0)) return a;
    if (va && feq(*va, 0.0)) return neg(b);  // 0 - x -> Neg(x)
    // a - (-b) -> a + b. Hit by the canonical objective form, whose rearranged
    // remainder is a Neg: `-45 - Neg(sum)` becomes `-45 + sum`.
    if (nodes_[b].kind == Kind::Neg) return add(a, nodes_[b].args[0]);
    return binary_node(Kind::Sub, {a, b}, nodes_[a].line);
  }

  auto mul(int a, int b) -> int
  {
    auto va = as_num(a), vb = as_num(b);
    if (va && vb) return num(*va * *vb, nodes_[a].line);
    if (vb && feq(*vb, 1.0)) return a;
    if (va && feq(*va, 1.0)) return b;
    if (vb && feq(*vb, -1.0)) return neg(a);
    if (va && feq(*va, -1.0)) return neg(b);
    return binary_node(Kind::Mul, {a, b}, nodes_[a].line);
  }

  auto div(int a, int b) -> int
  {
    auto va = as_num(a), vb = as_num(b);
    if (va && vb && !feq(*vb, 0.0)) return num(*va / *vb, nodes_[a].line);
    if (vb && feq(*vb, 1.0)) return a;
    if (vb && feq(*vb, -1.0)) return neg(a);
    return binary_node(Kind::Div, {a, b}, nodes_[a].line);
  }

  /// Literal value of node `i`, if it is one.
  auto as_num(int i) const -> std::optional<double>
  {
    if (nodes_[i].kind == Kind::Num) return nodes_[i].value;
    return std::nullopt;
  }

  /**
   * @brief Rewrite references to fixed symbols into literals, in ONE pass.
   *
   * `literal[s]` holds the value for symbol `s`, or nullopt to leave it alone.
   * Var nodes are leaves, so this is an in-place edit with no structural
   * consequences.
   *
   * The single pass is not a micro-optimisation: MINLPLib's acopf instances fix
   * over 14000 variables, so a pass per symbol is quadratic in the model and
   * takes minutes on inputs that otherwise parse in under a second.
   */
  void substitute_literals(std::vector<std::optional<double>> const& literal)
  {
    for (auto& n : nodes_) {
      if (n.kind != Kind::Var) continue;
      if (static_cast<std::size_t>(n.sym) >= literal.size()) continue;
      auto const& v = literal[static_cast<std::size_t>(n.sym)];
      if (!v) continue;
      n.kind = Kind::Num;
      n.value = *v;
      n.sym = -1;
    }
    fold_.clear();
    contains_.clear();
  }

  /**
   * @brief Per-node flag: does this node's subtree reference `sym`?
   *
   * Built by a forward sweep and extended incrementally as nodes are appended.
   * Not recursive: MINLPLib contains single expressions with hundreds of
   * thousands of terms, and a post-order recursion over one of those overflows
   * the stack.
   */
  auto contains_mask(int sym) const -> std::vector<char> const&
  {
    if (contains_sym_ != sym) {
      contains_sym_ = sym;
      contains_.clear();
    }
    for (std::size_t i = contains_.size(); i < nodes_.size(); ++i) {
      Node const& n = nodes_[i];
      char found = (n.kind == Kind::Var && n.sym == sym) ? 1 : 0;
      for (int a : n.args) {
        if (contains_[static_cast<std::size_t>(a)] != 0) { found = 1; break; }
      }
      contains_.push_back(found);
    }
    return contains_;
  }

  auto contains(int i, int sym) const -> bool
  {
    return contains_mask(sym)[static_cast<std::size_t>(i)] != 0;
  }

  /**
   * @brief Per-node constant value, present iff the subtree has no variable.
   *
   * Same forward-sweep, incrementally-extended discipline as contains_mask.
   * Used by split() (to recognise a constant multiplier) and by the lowering
   * (to emit a whole constant subtree as one Const node, which also guarantees
   * no unary Op is ever applied directly to a Const -- something GraphBuilder
   * rejects).
   */
  auto folds() const -> std::vector<std::optional<double>> const&
  {
    for (std::size_t i = fold_.size(); i < nodes_.size(); ++i) {
      fold_.push_back(compute_fold(static_cast<int>(i)));
    }
    return fold_;
  }

  auto fold(int i) const -> std::optional<double>
  {
    return folds()[static_cast<std::size_t>(i)];
  }

private:
  auto push(Node n) -> int
  {
    nodes_.push_back(std::move(n));
    return static_cast<int>(nodes_.size()) - 1;
  }

  auto binary_node(Kind k, std::vector<int> args, int line) -> int
  {
    Node n;
    n.kind = k;
    n.args = std::move(args);
    n.line = line;
    return push(n);
  }

  /// Value of node `i` from its children's already-computed values. Only ever
  /// called by folds(), in increasing index order, so every child is present.
  auto compute_fold(int i) const -> std::optional<double>
  {
    Node const& n = nodes_[i];
    if (n.kind == Kind::Num) return n.value;
    if (n.kind == Kind::Var) return std::nullopt;

    std::vector<double> a;
    a.reserve(n.args.size());
    for (int c : n.args) {
      auto const& v = fold_[static_cast<std::size_t>(c)];
      if (!v) return std::nullopt;
      a.push_back(*v);
    }

    switch (n.kind) {
      case Kind::Add: return a[0] + a[1];
      case Kind::Sub: return a[0] - a[1];
      case Kind::Mul: return a[0] * a[1];
      case Kind::Div: return feq(a[1], 0.0) ? std::nullopt : std::optional{a[0] / a[1]};
      case Kind::Neg: return -a[0];
      case Kind::Call: return fold_call(n.func, a);
      default: return std::nullopt;
    }
  }

  static auto fold_call(Func f, std::vector<double> const& a)
      -> std::optional<double>
  {
    switch (f) {
      case Func::Sqr:   return a[0] * a[0];
      case Func::Sqrt:  return a[0] < 0.0 ? std::nullopt : std::optional{std::sqrt(a[0])};
      case Func::Exp:   return std::exp(a[0]);
      case Func::Log:   return a[0] <= 0.0 ? std::nullopt : std::optional{std::log(a[0])};
      case Func::Log10: return a[0] <= 0.0 ? std::nullopt : std::optional{std::log10(a[0])};
      case Func::Log2:  return a[0] <= 0.0 ? std::nullopt : std::optional{std::log2(a[0])};
      case Func::Abs:   return std::fabs(a[0]);
      case Func::Sin:   return std::sin(a[0]);
      case Func::Cos:   return std::cos(a[0]);
      case Func::Tanh:  return std::tanh(a[0]);
      case Func::Min:   return std::fmin(a[0], a[1]);
      case Func::Max:   return std::fmax(a[0], a[1]);
      case Func::Power: {
        double e = a[1];
        // floor(e) < e (rather than !=) is "e is not already integer-valued" --
        // sidesteps -Wfloat-equal, same idiom as Problem::validate() in dag.hpp.
        if (std::floor(e) < e && a[0] <= 0.0) return std::nullopt;
        return std::pow(a[0], e);
      }
    }
    return std::nullopt;
  }

  std::vector<Node> nodes_;
  mutable std::vector<std::optional<double>> fold_;
  mutable std::vector<char> contains_;
  mutable int contains_sym_ = -2;
};

// ---------------------------------------------------------------------------

enum class Rel { E, L, G, N };

struct Symbol {
  std::string name;  // original spelling, for reporting
  int decl_line = 0;
  double lo = -std::numeric_limits<double>::infinity();
  double up = std::numeric_limits<double>::infinity();
  bool fixed = false;
  double fx = 0.0;
  bool has_level = false;
  double level = 0.0;
  cuminlp::dag::VarKind kind = cuminlp::dag::VarKind::Continuous;
  int integral_line = 0;  // where kind was last set away from Continuous,
                          // which is often a later re-declaration than
                          // decl_line; meaningless while kind == Continuous
  bool is_objvar = false;
  bool eliminated = false;  // objective variable, substituted away
};

struct Equation {
  std::string name;
  int line = 0;
  Rel rel = Rel::E;
  int lhs = -1;
  int rhs = -1;
  bool consumed = false;  // used as the objective's defining equation
};

struct Model {
  Ast ast;
  std::vector<Symbol> symbols;
  std::unordered_map<std::string, int> symbol_index;  // folded name -> index
  std::vector<Equation> equations;
  int objvar = -1;
  bool maximise = false;
  bool have_solve = false;
};

}
