# Design Doc: `.gms` File Frontend

**Status**: Implemented. Phase 4 measurements in §9.1; they supersede the
guesses this plan was written with.
**Scope**: A module that reads an optimisation problem from a MINLPLib-style
GAMS scalar file and produces a `cuminlp::dag::Problem<T>`. Frontend only —
everything downstream (graph replay, search) is unchanged and unaware this
exists.
**Last updated**: 28 July 2026

---

## 1. Format choice: `.gms`, not `.ams`

Both are offered by MINLPLib and both are *scalar* (sets, indices and loops
already expanded by the generator), so neither requires implementing a real
modelling language. Recommend `.gms`:

- **Smaller grammar.** GAMS Convert scalar output is a fixed, mechanical
  vocabulary: a handful of declaration statements, one equation-definition
  form, one attribute-assignment form, one `Solve` statement. `.ams` wraps
  every identifier in a multi-attribute AIMMS block (`VARIABLE: identifier:
  x1; range: [0, inf];`) nested inside `DECLARATION SECTION` / `MATHEMATICAL
  PROGRAM` sections — more syntactic surface for the same information.
- **The expression sublanguage is the same work either way.** Infix
  arithmetic with the same function names. That is the bulk of the parser, so
  the container format is the only real differentiator.
- **A free correctness oracle.** The hand-coded instances already in this repo
  (`source/power_series.cu` = ex4_1_2, `source/morse_cluster_energy.cu` =
  ex8_6_2) cite MINLPLib pages whose `.gms` is the canonical listing, so the
  parser can be tested against DAGs we already trust (§8).

`.ams` stays possible later: the AST and lowering stages (§4.3, §5, §6) are
format-agnostic, so an AIMMS reader would only replace the lexer and statement
parser.

### 1.1 The one hard precondition

This parses the **GAMS scalar format**, not GAMS. Files containing `Set`,
`Sets`, `Alias`, `Table`, `Parameter`, `loop`, `sum(`, `prod(`, or indexed
identifiers (`x(i)`) are out of scope and must fail fast with a message saying
so, rather than mis-parsing. MINLPLib's published `.gms` is scalar, but the
guard is cheap and prevents silent nonsense on a hand-written model.

---

## 2. Architecture

```
  .gms text
      │
      ▼
  [ lexer ]  case-folded tokens, comment/dollar-line stripping
      │
      ▼
  [ statement parser ] ──▶ declarations, equation defs, bound assignments,
      │                    solve statement   (all stored as a private AST,
      │                    nothing touches Problem yet)
      ▼
  [ semantic pass ]  bound resolution, objective-variable elimination (§6),
      │              discrete-variable rejection
      ▼
  [ lowering ]  ─────────▶ dag::Problem<T>  +  names, warnings, initial point
```

The **two-pass split is mandatory, not stylistic**. `Problem::var(lb, ub)`
fixes a variable's box at creation time and assigns `var_index` from
`box_bounds.size()`, but a `.gms` file states bounds (`x1.lo = 1;`) *after*
the equations that use `x1`, and names the objective variable (in `Solve`)
last of all. So the parser must build its own AST first and emit into
`ExprDAG` only once every fact is known.

The private AST is also what makes §6 tractable: rearranging an equation and
folding constants are trivial on a small tagged tree and awkward on an
append-only, topologically-ordered `ExprDAG`. Everything is settled in the AST;
`DAGNode`s are emitted once, in final form.

### 2.1 Files

Proportionate to the repo's current size — three files, not a subdirectory
tree:

| File | Contents |
| --- | --- |
| `include/cuminlp/gams.hpp` | Public API only (§3). No parser internals. |
| `source/gams_ast.hpp` | Internal: token type, AST node, symbol table. |
| `source/gams_frontend.cpp` | Lexer, parsers, semantic pass, lowering. |

### 2.2 Build integration

```cmake
add_library(gams_frontend source/gams_frontend.cpp)
add_library(cuminlp::gams ALIAS gams_frontend)
target_link_libraries(gams_frontend PUBLIC cuminlp::lib)
target_compile_features(gams_frontend PUBLIC cxx_std_20)
```

Deliberately **CXX, not CUDA** — a `.cpp`, no `CUDA_ARCHITECTURES` property.
Verified: `dag.hpp` compiles under plain `g++ -std=c++20` against
`lib/cuinterval/include`, so the frontend needs no nvcc, compiles fast, and is
testable on a machine with no GPU (unlike every existing test target — see
`TESTING.md`). No external dependencies. `Problem<T>` is a template, so the
`.cpp` ends with explicit instantiation for `double` (and `float` if wanted).

### 2.3 No changes to `dag.hpp` (revised during implementation)

The plan offered two options: add the missing `Expr` friends (`log`, `sin`,
`cos`, `tanh`, `abs`, `min`, `max`), or emit through `problem.graph.emit(...)`
directly. **As built, the lowering emits directly and `dag.hpp` is untouched.**
The lowering works in node ids rather than `Expr` values anyway (§5, the emit
sweep), and leaving the shared header alone keeps this frontend clear of
concurrent work on it. The one thing `Expr` would have given for free —
`operator*` collapsing `a * a` into `Op::Sqr` — is reproduced in
`emit_binary`.

---

## 3. Public API

```cpp
namespace cuminlp::gams {

enum class Sense { Minimise, Maximise };

struct ParseOptions {
  // Substituted for a variable with no finite bound. The interval evaluator
  // returns useless bounds on an unbounded box and the search driver would
  // partition it forever, so free variables must be given *some* box.
  double default_bound = 1e6;

  // Emit `Problem::fixed()` (a Const node, no search dimension) for `x.fx =`
  // variables, matching the symmetry-breaking usage in
  // source/morse_cluster_energy.cu. Off => a degenerate [v, v] box.
  bool fold_fixed_to_const = true;

  // Discrete variables have no representation in Problem (§5.3).
  bool reject_discrete = true;
};

struct Diagnostic { int line; std::string message; };

struct ParseError : std::runtime_error {
  int line;
};

template<typename T>
struct ParsedModel {
  dag::Problem<T> problem;

  Sense sense = Sense::Minimise;   // objective in `problem` is ALREADY negated
                                   // if Maximise; this is for reporting only
  std::vector<std::string> var_names;         // index == var_index in box_bounds
  std::vector<std::string> constraint_names;  // index == index in constraints
  std::vector<T> initial_point;               // from `x.l =`; empty if absent
  std::vector<Diagnostic> warnings;
};

template<typename T>
auto parse(std::string_view source, ParseOptions const& = {}) -> ParsedModel<T>;

template<typename T>
auto parse_file(std::filesystem::path const&, ParseOptions const& = {})
    -> ParsedModel<T>;

}
```

Design notes:

- **`ParsedModel`, not bare `Problem`.** The names, the initial point and the
  warning list are exactly what a benchmark harness needs and none of them
  belong in the IR. `var_names[i]` giving the GAMS name for `box_bounds[i]` is
  what makes a solver trace readable.
- **Warnings vs errors.** Anything that changes the mathematics silently
  (a substituted default bound, an `**` domain widening, a dropped `=N=`
  equation) is a `Diagnostic`, never silent. Anything we cannot represent is a
  `ParseError` with a line number — never a partially-built `Problem`.
- **`initial_point`** is carried but unused today; it is free to capture and is
  the obvious seed for the sampling in `GraphDriver`.

---

## 4. Parsing stages

### 4.1 Lexer

- **Case-insensitive.** GAMS folds case for both keywords and identifiers, so
  `X1` and `x1` are one symbol. Fold to lower case at token creation and key
  the symbol table on the folded form; keep the original spelling for
  `var_names`.
- **Line comments**: `*` in column 1 kills the line. (`*` elsewhere is
  multiplication — column position is the discriminator.)
- **Dollar-control lines**: any line whose first non-space character is `$` is
  skipped, with `$onText` / `$offText` handled as a block skip. Covers
  `$offlisting`, `$if NOT '%gams.u1%' == '' $include ...`, `$if not set NLP
  $set NLP NLP`.
- **`%...%` macro tokens** (e.g. `Solve m using %NLP% minimizing x2;`): treat
  as an opaque token; the model-type field is not needed for lowering.
- **Numbers**: `strtod` / `std::from_chars` for full round-trip of literals
  like `0.333333333333333` and `1e+299`. Also recognise the GAMS special
  values `INF`, `-INF`, `EPS`, `NA`, `UNDF` — these appear in bound
  assignments and must not be lexed as identifiers.
- **Statements terminate at `;`** and freely span lines; the lexer is
  line-agnostic except for the two column-1 rules above. Track line numbers on
  every token for diagnostics.

### 4.2 Statement parser

Recognised statement forms — this is the complete vocabulary:

| Form | Action |
| --- | --- |
| `[Free\|Positive\|Negative\|Binary\|Integer] Variables a,b,c;` | Declare, record domain default |
| `Equations e1,e2;` | Declare equation names |
| `e1..  <expr> =E= <expr>;` (also `=L=`, `=G=`, `=N=`) | Parse both sides into AST |
| `x1.lo = <num>;` / `.up` / `.fx` / `.l` | Record bound / initial level |
| `x1.m =`, `e1.m =`, `m.limrow=0;` etc. | Ignore (marginals, model attrs) |
| `Model m / all /;` | Ignore |
| `Solve m using NLP minimizing x2;` | Record sense + objective variable |
| `Sos1 Variables`, `Semicont Variables`, `Table`, `Set`, `loop`, ... | `ParseError` (§1.1, §5.3) |

Domain defaults, applied before any explicit `.lo`/`.up`:

| Declaration | Default box |
| --- | --- |
| `Variables` / `Free Variables` | `(-inf, +inf)` |
| `Positive Variables` | `[0, +inf)` |
| `Negative Variables` | `(-inf, 0]` |
| `Binary Variables` | `{0, 1}` — discrete, see §5.3 |
| `Integer Variables` | `[0, +inf)` integral — discrete, see §5.3 |

Unknown assignment targets (a `.m` on an equation, a model attribute) are
skipped rather than erroring: they carry no information we consume, and being
strict here would reject valid files over solver hints.

### 4.3 Expression parser

Recursive-descent / Pratt, precedence low to high:

```
expr   := term (('+' | '-') term)*
term   := unary (('*' | '/') unary)*
unary  := ('-' | '+')* power
power  := atom ('**' unary)?          -- right-associative
atom   := number | ident | ident '(' expr (',' expr)* ')' | '(' expr ')'
```

Two GAMS-specific precedence facts to get right, both easy to get wrong:

- **`**` binds tighter than unary minus**: `-x**2` is `-(x**2)`, not `(-x)**2`.
- **`**` is right-associative** and its exponent may itself be signed:
  `2**3**2` is `2**(3**2)`; `x**-1` is legal.

The AST node is deliberately small and mutable, because §6 rewrites it:

```cpp
struct Node {                 // in source/gams_ast.hpp
  Kind kind;                  // Num, Var, Add, Sub, Mul, Div, Neg, Call
  double value = 0;           // Num
  int sym = -1;               // Var: symbol-table index
  Func func = Func::None;     // Call
  std::vector<int> args;      // child node indices
  int line = 0;
};
// AST is a flat std::vector<Node>; children referenced by index, so
// substitution and rearrangement are index rewrites, never pointer surgery.
```

### 4.4 Operator mapping

Direct, no rewriting:

| GAMS | `Op` |
| --- | --- |
| `+ - * /`, unary `-` | `Add` `Sub` `Mul` `Div` `Neg` |
| `sqr` `sqrt` `exp` `log` `abs` | `Sqr` `Sqrt` `Exp` `Log` `Abs` |
| `sin` `cos` `tanh` | `Sin` `Cos` `Tanh` |
| `min(a,b,...)` `max(a,b,...)` | `Min` / `Max`, folded pairwise left to right |
| `power(x, n)`, integral `n` | `IPow` (payload `int_exp`) |

Cheap rewrites (no new ops needed):

| GAMS | Emitted as |
| --- | --- |
| `log10(x)` | `Mul(Log(x), Const(1/ln 10))` |
| `log2(x)` | `Mul(Log(x), Const(1/ln 2))` |
| constant subtrees | folded in the AST, emitted as one `Const` |

Unsupported today — hard `ParseError` naming the function and line, so the
corpus run (§9, Phase 3) reports exactly how much coverage each would buy:
`tan`, `arctan`, `errorf`, `sign`, `signpower`, `ceil`, `floor`, `mod`, `div`,
`centropy`, `sinh`, `cosh`. Refusing beats an inexact rewrite: `sinh` via
`(exp(x) - exp(-x))/2` evaluates correctly but loses interval tightness by
splitting one dependency into two, which is precisely the kind of silent
bound degradation this solver exists to avoid.

#### 4.4.1 The `**` decision (needs an explicit call)

`x**c` is the one genuinely ambiguous mapping.

- GAMS defines `**` via `exp(c * log(x))`, so it **requires `x > 0`** even when
  `c` is integral. That is exactly why GAMS Convert emits `sqr(x)` and
  `power(x, n)` explicitly where it can — a surviving `**` is usually a real
  non-integer power.
- `c` an integer literal → **emit `IPow`**, and warn. This widens the domain to
  `x <= 0` relative to strict GAMS semantics. Justification: it matches the
  mathematical intent of the underlying model, and it is what a bounding solver
  wants; a literal reading would make the objective undefined over half of a
  box the search legitimately wants to explore. The warning keeps it honest.
- `c == 0.5` → **`Op::Sqrt`**. Not in the original plan, and not a lossy
  rewrite: sqrt is the same function with the same interval enclosure, in one
  op. MINLPLib writes Euclidean norms as `(...)**0.5`, and ex8_6_2 — the
  marquee test case of §8 — is one of them, so without this the plan's own
  headline instance would have been rejected.
- `c` otherwise non-integer → **needs a new op**. Recommendation: add a unary
  `Op::Pow` whose real exponent lives in `payload.constant`. This fits the
  existing `DAGNodePayload` union with **zero changes to its layout**, needs one
  `wire_unary`-style case in `graph_replay.cuh`, and keeps the natural interval
  enclosure. The alternative — rewriting to `Exp(Mul(Const(c), Log(x)))` — is
  representable today but strictly weaker on bounds and undefined at `x = 0`.
  Phase 4 (§9); until then, `ParseError`.
- `c` a non-constant expression (`x**y`) → `ParseError`. Vanishingly rare in
  MINLPLib and not worth speculative support.

---

## 5. Lowering

### 5.1 Variables

- Emit in **declaration order**, so `var_index` matches the file's order and
  `var_names[i]` is a stable, reportable mapping.
- Resolve each box as: domain default, overridden by `.lo` / `.up`, overridden
  by `.fx`.
- **Infinite bounds** → substitute `±ParseOptions::default_bound` and warn,
  listing the affected variables. `Problem::var()`'s `lowest()`/`max()` default
  is representable but useless to an interval evaluator, and many MINLPLib NLPs
  declare free variables. This is the single most likely source of "it parsed
  but the solver does nothing sensible", so the warning matters.
- **`x.fx = v`** → the variable is rewritten to the literal `v` **in the AST**
  when `fold_fixed_to_const`, rather than calling `Problem::fixed(v)` as
  planned. Same outcome (no search dimension, a `Const` in the DAG) plus two
  things the planned version would not have got: constant subtrees containing
  fixed variables now fold away entirely, and — the reason this is not merely
  an optimisation — a fixed variable can no longer end up as the direct
  operand of a unary op, which `GraphBuilder` rejects outright ("unary op
  applied directly to an Op::Const"). `sqr(x1)` with `x1` fixed would have hit
  exactly that.
- **The objective variable is never lowered** — see §6.

### 5.2 Constraints

`ConstraintRef` is `{root_id, Cmp{LE, EQ}, T rhs}` — one-sided, scalar RHS.
GAMS relations map in without extending it:

| GAMS | Emitted |
| --- | --- |
| `L =E= R`, `R` constant | `root = L`, `EQ`, `rhs = R` |
| `L =E= R`, `R` non-constant | `root = L - R`, `EQ`, `rhs = 0` |
| `L =L= R` | as above with `LE` |
| `L =G= R` | swap sides, then treat as `R =L= L` |
| `L =N= R` | dropped, with a warning (no relation asserted) |

Folding a constant RHS out of the DAG rather than emitting `Sub(L, Const)` is a
real saving: it removes one graph node per constraint, and the fold happens in
the frontend AST, so nothing downstream sees a rewrite.

### 5.3 Discrete variables (future work)

`Binary` and `Integer` declarations are a `ParseError` by default. `Problem`
has no integrality field and the search driver has no integer branching, so
accepting them would produce a wrong answer rather than a slow one. Lifting
this needs both a `std::vector<uint8_t>` parallel to `box_bounds` *and*
branching support in the driver — a search-driver change, tracked as future
work, out of scope for this module. The parser should nonetheless *record* the
integrality it saw, so that when the driver gains support the frontend needs
only to stop erroring.

---

## 6. Objective-variable elimination

This is the one stage with real algorithmic content, so it gets its own
section. **It is easier than it looks**, because the only case that has to be
solved is *linear in one variable*.

### 6.1 The problem

GAMS Convert models have no objective *expression* — they have an objective
*variable* plus a defining equation:

```
Solve m using NLP minimizing objvar;
e1..  - (0.5*x1 + x1*x2) + objvar =E= 0;
```

`Problem::set_objective` wants an `Expr`, so `objvar` must be eliminated. Doing
it properly matters: leaving `objvar` as a `Var` and adding `e1` as an equality
constraint is correct, but it hands the search driver an extra dimension to
branch on that is fully determined by the others, and turns an unconstrained
problem (like both instances already in this repo) into a constrained one.

### 6.2 The algorithm: linear coefficient extraction

Everything reduces to one recursive function over the AST. Define

```
split(node, v) -> optional<(double coeff, int remainder)>
```

with the contract **`node ≡ coeff * v + remainder`, and `remainder` does not
contain `v`**. Returning `nullopt` means "not linear in `v`", which is a
detection result, not a failure to be worked around.

```cpp
// Helpers, both trivial recursive walks over the flat AST:
//   contains(n, v)   -> bool
//   fold(n)          -> optional<double>   // value iff subtree has no Var leaf

std::optional<Split> split(int n, int v) {
  Node const& x = ast[n];
  if (!contains(n, v)) return Split{0.0, n};          // whole subtree is remainder

  switch (x.kind) {
    case Var:  return Split{1.0, num(0.0)};            // x.sym == v here
    case Neg:  { auto a = split(x.args[0], v); if (!a) return {};
                 return Split{-a->coeff, neg(a->rem)}; }
    case Add:  { auto a = split(x.args[0], v), b = split(x.args[1], v);
                 if (!a || !b) return {};
                 return Split{a->coeff + b->coeff, add(a->rem, b->rem)}; }
    case Sub:  { auto a = split(x.args[0], v), b = split(x.args[1], v);
                 if (!a || !b) return {};
                 return Split{a->coeff - b->coeff, sub(a->rem, b->rem)}; }
    case Mul:  { // exactly one side may contain v, and the other must be constant
                 int cv = contains(x.args[0], v) ? 0 : 1;
                 if (contains(x.args[cv], v)) return {};        // v * v -> nonlinear
                 auto k = fold(x.args[cv]); if (!k) return {};  // v * x1 -> nonlinear
                 auto a = split(x.args[1 - cv], v); if (!a) return {};
                 return Split{*k * a->coeff, mul(x.args[cv], a->rem)}; }
    case Div:  { if (contains(x.args[1], v)) return {};         // v in denominator
                 auto k = fold(x.args[1]); if (!k || *k == 0) return {};
                 auto a = split(x.args[0], v); if (!a) return {};
                 return Split{a->coeff / *k, div(a->rem, x.args[1])}; }
    default:   return {};   // v under exp/log/sqr/ipow/... -> nonlinear
  }
}
```

The `if (!contains(...)) return {0, n}` guard at the top is what keeps this
short: every subtree free of `v` is handled once, generically, so the switch
only ever deals with the cases where `v` actually appears.

Then, for an `=E=` equation `L =E= R`:

```
(cL, rL) = split(L, objvar)          // fail -> not a defining equation
(cR, rR) = split(R, objvar)
c = cL - cR                          // |c| < eps -> objvar cancels, not defining
objective = (rR - rL) / c
```

On the canonical Convert form `-(P) + objvar =E= 0`: `split(L)` gives
`(1, Neg(P))`, `split(R)` gives `(0, 0)`, so `c = 1` and the objective is
`(0 - Neg(P)) / 1`. A small AST peephole (drop `x + 0`, `x - 0`, `x * 1`,
`x / 1`, collapse `Neg(Neg(x))`) reduces that to `P` exactly, so the common
case emits zero junk nodes. Without the peephole it would emit three.

Total: `split` plus `contains` plus `fold` plus the peephole is roughly 120
lines, and none of it is subtle.

### 6.3 Driving it

1. From `Solve`, take the objective variable and the sense.
2. Run §6.2 over every `=E=` equation mentioning `objvar`. Take the first that
   succeeds as the **defining equation**; warn if more than one does.
3. Rearrange to get the objective AST `P`. Drop that equation from the
   constraint list. Mark `objvar` eliminated so it is never lowered as a `Var`.
4. `maximizing` → wrap the objective root in `Op::Neg` and set
   `ParsedModel::sense = Maximise` so the caller can flip the number it prints.
   The driver only minimises.

**`objvar` in *other* equations costs nothing.** Because every function shares
one `ExprDAG`, substitution is free at emit time: lower `P` once, keep its root
id, and have the lowering map any remaining `Var(objvar)` reference to that
same id. No subtree duplication, no extra nodes, and no restriction that
`objvar` appear in exactly one equation. Do the substitution at *lowering*
time, not by rewriting the AST — an AST substitution would emit `P` once per
occurrence.

**Surviving bounds.** If the eliminated `objvar` carried a finite non-default
`.lo`/`.up`, that bound is a genuine constraint on the objective and must not
be dropped: emit it as an ordinary constraint on `P` (`P <= up`, and `-P <= -lo`
via §5.2's `=G=` handling). Easy to forget, rare but real.

### 6.4 When it doesn't apply

| Situation | Response |
| --- | --- |
| No `=E=` equation is linear in `objvar` | Fallback (below) |
| `objvar` appears nonlinearly (`sqr(objvar)`) | Fallback |
| Defining equation is `=L=` / `=G=` | Fallback + warning. (Solving `min objvar` s.t. `objvar >= f(x)` is equivalent to `min f(x)`, so this *could* be eliminated for the matching sense direction — not worth building until an instance needs it.) |
| No `Solve` statement | `ParseError` |

**Fallback**: keep `objvar` as a real `Var` with its declared bounds, add all
its equations as ordinary constraints, and set the objective to `objvar`.
Mathematically identical, costs one search dimension, emits a warning. Having
this path is what stops an unusual instance from being a hard failure, and it
means §6.2 is allowed to be strict — anything it declines degrades gracefully
rather than needing ever-more-clever rearrangement.

---

## 7. Worked example

`ex4_1_2.gms` (already hand-coded in `source/power_series.cu`) reduces to:

```
Variables  x1,x2;
Equations  e1;
e1..  -(1.666666666*power(x1,3) + ... + 0.8*power(x1,50) + sqr(x1) + x1) + x2 =E= 0;
x1.lo = 1;  x1.up = 2;
Solve m using NLP minimizing x2;
```

Lowering: `x1` → `problem.var(1, 2)` (index 0); `x2` is the objective variable,
eliminated via §6 (`c = 1`, objective `= P`), never lowered;
`power(x1, k)` → `IPow` with `int_exp = k`; `sqr(x1)` → `Sqr`; `e1` dropped
after substitution, leaving zero constraints. Result: 1 variable, no
constraints, objective root over ~50 `IPow` nodes — structurally identical to
the hand-written builder, which is the point.

---

## 8. Testing

The strongest available oracle is already in the repo: **two MINLPLib
instances are hand-coded as programmatic-frontend DAGs**
(`source/power_series.cu` = ex4_1_2, `source/morse_cluster_energy.cu` =
ex8_6_2). Parsing their `.gms` files must produce a DAG that agrees with the
hand-built one.

Layers:

1. **Host DAG evaluator** (`test/source/dag_eval.hpp`, ~40 lines): plain
   `double` post-order walk over `ExprDAG`. Written first — every later
   assertion needs it, and it is independently useful as the CPU oracle for
   validating directed rounding in the GPU kernels. Reusable well beyond this
   module.
2. **Expression unit tests** (Catch2, already vendored in `test/CMakeLists.txt`):
   parse an expression string, evaluate at a point, compare to the expected
   value. Covers precedence (`-x**2`, `2**3**2`), case folding, comments,
   dollar lines, number formats, each mapped function.
3. **Elimination unit tests** (§6, the highest-risk stage): feed `split` the
   shapes that matter — `-(P) + v =E= 0`, `v - P =E= 0`, `2*v + P =E= 3`,
   `P =E= v`, and the rejects `sqr(v) =E= P`, `v*x1 =E= P`, `P/v =E= 1`.
   Assert the recovered objective evaluates equal to `P`, and that the rejects
   take the §6.4 fallback rather than erroring.
4. **Equivalence tests**: check `ex4_1_2.gms` and `ex8_6_2.gms` into
   `test/data/gams/`; parse each, evaluate both the parsed DAG and the
   hand-built DAG at N pseudo-random points in the box, assert agreement to
   tolerance. This catches operator mapping, objective substitution and
   variable ordering in one assertion. Also assert node counts are within a
   small factor of the hand-built DAG — that is what catches peephole
   regressions leaking junk nodes.
5. **Diagnostic tests**: an unsupported function, a `Binary Variables`
   declaration, a non-scalar `Set` statement and a malformed statement each
   raise `ParseError` with the correct line number.
6. **End-to-end (GPU, opt-in)**: parse `ex8_6_2.gms`, hand it to
   `GraphDriver`, assert the same incumbent as `graph_morse`. Gated like the
   existing CUDA tests.

Layers 1–5 need no GPU, so unlike every current test target they run
anywhere — worth stating in `TESTING.md`.

---

## 9. Phasing

| Phase | Work | Est. | Done when |
| --- | --- | --- | --- |
| 0 | Host DAG evaluator + canonical DAG-dump helper, in test support | 0.5 d | Evaluates the hand-built ex4_1_2 DAG correctly |
| 1 | Lexer, expression parser, AST; expression-level tests only, no `Problem` | 2 d | Test layer 2 green |
| 2 | `split`/`contains`/`fold` + peephole + elimination driver (§6) | 0.5 d | Test layer 3 green |
| 3 | Statement parser, symbol table, lowering, public API, CMake target | 1.5 d | Test layers 4–5 green |
| 4 | Run over a downloaded MINLPLib `.gms` corpus; emit a coverage report (parsed / unsupported-op / discrete / non-scalar / elimination-fallback / other) | 0.5 d | A table saying what fraction parses and what each missing feature would buy |
| 5 | Driven by Phase 4's numbers, in priority order: `Op::Pow` (real exponent, §4.4.1), integrality in `Problem` + search branching (§5.3), remaining transcendentals | — | — |

Phase 2 is deliberately sequenced before the statement parser: it is the part
with real algorithmic risk, it is testable against hand-written ASTs with no
lexer involved, and if `split` turns out to need more cases than §6.2 lists,
that is much better discovered on day three than on day five.

Phase 4 is the decision point: it converts "which ops should I add" from a
guess into a measurement, and the coverage number is quotable in a write-up.

### 9.1 Phase 4 results (all 1634 MINLPLib `.gms` instances)

From `gams_report`, 55.7 s wall for the whole library:

| Outcome | Count | Share |
| --- | ---: | ---: |
| **Parsed** | **532** | 32.6% |
| — objective variable eliminated | 508 | |
| — took the §6.4 fallback | 24 | |
| — needed a default bound (§5.1) | 276 | |
| Rejected: discrete variables | 1033 | 63.2% |
| Rejected: non-integer exponent | 33 | 2.0% |
| Rejected: unsupported function | 25 | 1.5% |
| Rejected: not scalar format | 6 | 0.4% |
| Rejected: unrepresentable | 5 | 0.3% |

**The headline number is not 32.6%.** 1033 of the 1102 rejections are discrete
variables, which is a `Problem`/search-driver limitation, not a frontend one —
MINLPLib is mostly MINLP. Of the 601 instances with only continuous variables,
**532 parse, or 88.5%**. Every remaining rejection has a named cause; there is
no residual "syntax error" bucket.

What this changes about Phase 5's ordering:

1. **Integrality is worth more than every operator combined** — 1033 instances
   against 58 for all missing maths put together. It is also the more expensive
   item, since it needs branching in the search driver, not just a field on
   `Problem`.
2. **`Op::Pow` (§4.4.1) is the best value per unit of work**: 33 instances, one
   op, one `wire_unary` case. Exponents seen are scattered (0.6 in 7 instances,
   −0.24 in 6, then a long tail), so a real exponent is needed — no special
   case would cover them.
3. **`signpower` alone is 18 of the 25 function rejections**, and the rest are a
   thin tail (`errorf` 4, `mod` 1, `centropy` 1). If any function is added, it
   is that one.
4. **276 of 532 parsed instances lean on a default bound.** That is the largest
   *silent* quality issue in the set: they parse, then hand the search an
   arbitrary 10^6 box. Worth revisiting `ParseOptions::default_bound`, and worth
   remembering when reading any benchmark built from this corpus.

### 9.2 Two performance findings from the corpus run

Both were invisible on small instances and only appeared at library scale.

- **Fixed-variable substitution must be one pass.** `acopf_case6468rte_qcqp`
  fixes 14150 variables; a pass over the node vector per fixed symbol made it
  quadratic — 45.9 s, versus 0.83 s for the single-pass version, with the
  largest instance going from a >100 s timeout to 8.9 s.
- **No recursive tree walks.** MINLPLib has single expressions with hundreds of
  thousands of terms, and post-order recursion over one of those overflows the
  stack (the first whole-corpus run died on SIGSEGV). `contains`, `fold`,
  `split` and the lowering are all linear sweeps instead, which the Ast's
  "child index < parent index" invariant (§4.3) makes straightforward: mark
  reachability backwards from the roots, emit forwards.

---

## 10. Risks

- **Non-scalar `.gms` in the wild.** Mitigated by the fail-fast guard (§1.1);
  the failure message should name GAMS Convert as the fix.
- **`**` domain semantics** (§4.4.1) — the one place we knowingly diverge from
  a literal reading of the source language. Warned, and documented here.
- **Free variables** (§5.1) — parses cleanly, then the solver flounders on an
  effectively unbounded box. Warned, with a tunable default.
- **Elimination fallback rate** (§6.4) — if a meaningful share of the corpus
  takes the fallback, every one of those instances silently gains a search
  dimension. Phase 4's report must count fallbacks separately from failures for
  exactly this reason.
- **Coverage unknown until Phase 4.** The op list in §4.4 is drawn from what
  MINLPLib scalar files typically use, not from a census. Phase 4 exists
  precisely to replace that assumption with a count, and Phase 5 is
  deliberately left unordered until it reports.
