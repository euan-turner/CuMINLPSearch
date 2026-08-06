#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "cuminlp/gams.hpp"

#include <catch2/catch_test_macros.hpp>

#include "autocorr_bern20_03_problem.hpp"
#include "cuminlp/model/problem.hpp"
#include "dag_eval.hpp"

using cuminlp::model::Expr;
using cuminlp::model::Problem;
using cuminlp::testing::evaluate_objective;
namespace gams = cuminlp::gams;

namespace
{

/// Wrap an expression in the shape GAMS Convert emits, so an expression can be
/// tested without writing a whole model by hand.
auto model_with(std::string const& expression) -> std::string
{
  return "Variables x1,x2,objvar;\n"
         "Equations e1;\n"
         "e1.. -(" + expression + ") + objvar =E= 0;\n"
         "x1.lo = -10; x1.up = 10;\n"
         "x2.lo = -10; x2.up = 10;\n"
         "Solve m using NLP minimizing objvar;\n";
}

/**
 * Parse `expression` and evaluate the resulting objective at (x1, x2).
 *
 * `0*x1` is appended so that a wholly constant expression still produces a
 * variable objective -- the parser rejects a constant objective, and rightly
 * so, since GraphBuilder cannot produce a Const root. The Ast peephole
 * deliberately does not fold `0 * x` (that identity is unsound for intervals),
 * so this term survives to the DAG and contributes exactly zero.
 */
auto eval(std::string const& expression,
          double x1 = 0.0,
          double x2 = 0.0) -> double
{
  auto parsed = gams::parse<double>(model_with("(" + expression + ") + 0*x1"));
  return evaluate_objective(parsed.problem, {x1, x2});
}

auto data_file(char const* name) -> std::string
{
  return std::string(CUMINLP_GAMS_TEST_DATA) + "/" + name;
}

constexpr double kTol = 1e-9;

auto close(double a, double b, double tol = kTol) -> bool
{
  double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
  return std::fabs(a - b) <= tol * scale;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layer 2: lexing and expression parsing
// ---------------------------------------------------------------------------

TEST_CASE("arithmetic and precedence", "[gams][expr]")
{
  CHECK(close(eval("1 + 2*3"), 7.0));
  CHECK(close(eval("(1 + 2)*3"), 9.0));
  CHECK(close(eval("10 - 2 - 3"), 5.0));  // left-associative
  CHECK(close(eval("100/10/2"), 5.0));  // left-associative
  CHECK(close(eval("2*x1 + 3*x2", 5.0, 7.0), 31.0));
}

TEST_CASE("unary minus binds looser than **", "[gams][expr]")
{
  // -x**2 is -(x**2), not (-x)**2.
  CHECK(close(eval("-x1**2", 3.0), -9.0));
  CHECK(close(eval("(-x1)**2", 3.0), 9.0));
  // ** is right-associative: 2**3**2 == 2**9 == 512.
  CHECK(close(eval("2**3**2"), 512.0));
  // A signed exponent is legal.
  CHECK(close(eval("x1**-1", 4.0), 0.25));
}

TEST_CASE("functions map onto DAG ops", "[gams][expr]")
{
  CHECK(close(eval("sqr(x1)", 3.0), 9.0));
  CHECK(close(eval("sqrt(x1)", 9.0), 3.0));
  CHECK(close(eval("exp(x1)", 1.0), std::exp(1.0)));
  CHECK(close(eval("log(x1)", 2.0), std::log(2.0)));
  CHECK(close(eval("abs(x1)", -3.0), 3.0));
  CHECK(close(eval("sin(x1)", 1.0), std::sin(1.0)));
  CHECK(close(eval("cos(x1)", 1.0), std::cos(1.0)));
  CHECK(close(eval("tanh(x1)", 1.0), std::tanh(1.0)));
  CHECK(close(eval("power(x1,3)", 2.0), 8.0));
  CHECK(close(eval("min(x1,x2)", 3.0, 5.0), 3.0));
  CHECK(close(eval("max(x1,x2,7)", 3.0, 5.0), 7.0));
  // log10 / log2 are rewrites, not ops.
  CHECK(close(eval("log10(x1)", 1000.0), 3.0));
  CHECK(close(eval("log2(x1)", 8.0), 3.0));
}

TEST_CASE("x**0.5 lowers to sqrt", "[gams][expr]")
{
  // MINLPLib writes Euclidean norms this way; sqrt is the same function, so
  // this is a rename rather than an approximation.
  CHECK(close(eval("(sqr(x1) + sqr(x2))**0.5", 3.0, 4.0), 5.0));
}

TEST_CASE("x**c lowers to exp(c*log(x)) for a non-0.5, non-integer c",
          "[gams][expr]")
{
  // This is GAMS's own definition of `**`, not a widening or approximation
  // (see design/GAMS_FRONTEND.md): it needs x > 0, exactly as a
  // literal reading of `**` would. nvs09 (https://www.minlplib.org/nvs09.html)
  // is the MINLPLib instance that motivated this -- its objective has a
  // `product**0.2` term, hand-built in nvs09_problem.hpp as
  // exp(0.2*log(product)).
  CHECK(close(eval("x1**1.5", 4.0), std::pow(4.0, 1.5)));
  CHECK(close(eval("x1**0.2", 3.0), std::pow(3.0, 0.2)));
  CHECK(close(eval("x1**-0.5", 4.0), std::pow(4.0, -0.5)));
}

TEST_CASE("x**y lowers to exp(y*log(x)) when the exponent is not constant",
          "[gams][expr]")
{
  // A non-constant exponent used to be a ParseError (ErrorKind::
  // UnsupportedExponent); a corpus run over MINLPLib turned up real
  // instances of exactly this shape (e.g. contvar.gms's
  // `10**(-3.5 + 4.9*x/(1+x))`, hda.gms's `4**(0.0001 + 0.333*x1)`), so it
  // gets the same exp(y*log(x)) treatment as a non-integer constant
  // exponent, just with y as an ordinary DAG operand instead of a payload
  // constant.
  CHECK(close(eval("x1**x2", 2.0, 3.0), std::pow(2.0, 3.0)));
  CHECK(close(eval("10**(x1 + 0.5*x2)", 2.0, 3.0),
              std::pow(10.0, 2.0 + 0.5 * 3.0)));
}

TEST_CASE("GAMS is case-insensitive", "[gams][lex]")
{
  CHECK(close(eval("POWER(X1,2) + SQR(x1)", 3.0), 18.0));
}

TEST_CASE("column-1 star is a comment but an indented star is multiplication",
          "[gams][lex]")
{
  // Regression: a continuation line in Convert output routinely begins with an
  // indented `*`, which must not be read as a comment.
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      "Equations e1;\n"
      "* this whole line is a comment\n"
      "e1.. -(2\n"
      "     *x1) + objvar =E= 0;\n"
      "x1.lo = 0; x1.up = 10;\n"
      "Solve m using NLP minimizing objvar;\n");
  CHECK(close(evaluate_objective(parsed.problem, {3.0}), 6.0));
}

TEST_CASE("dollar-control lines are skipped", "[gams][lex]")
{
  auto parsed = gams::parse<double>(
      "$offlisting\n"
      "$ontext\n"
      "this text is not GAMS at all ((( ;;;\n"
      "$offtext\n"
      "Variables x1,objvar;\n"
      "Equations e1;\n"
      "e1.. -(x1) + objvar =E= 0;\n"
      "x1.lo = 0; x1.up = 1;\n"
      "$if NOT '%gams.u1%' == '' $include '%gams.u1%'\n"
      "Solve m using %NLP% minimizing objvar;\n");
  CHECK(close(evaluate_objective(parsed.problem, {0.5}), 0.5));
}

// ---------------------------------------------------------------------------
// Layer 3: objective-variable elimination
// ---------------------------------------------------------------------------

namespace
{

struct Eliminated
{
  bool eliminated;
  double value;  // objective evaluated at x1 = 3
  std::size_t vars;
};

auto try_eliminate(std::string const& equation) -> Eliminated
{
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      "Equations e1;\n"
      + equation + "\n"
      "x1.lo = -10; x1.up = 10;\n"
      "objvar.lo = -1e6; objvar.up = 1e6;\n"
      "Solve m using NLP minimizing objvar;\n");

  bool kept = false;
  for (auto const& name : parsed.var_names) {
    kept = kept || name == "objvar";
  }

  std::vector<double> point(parsed.problem.box_bounds.size(), 3.0);
  return {!kept,
          evaluate_objective(parsed.problem, point),
          parsed.problem.box_bounds.size()};
}

/// try_eliminate's sibling for the inequality path, which is sense-dependent
/// and so cannot be pinned to `minimizing` the way try_eliminate is. Also
/// leaves objvar unbounded, since the bound rules differ between the two paths
/// and get their own test below.
auto try_eliminate_ineq(std::string const& equations,
                        char const* sense = "minimizing",
                        char const* declare = "Equations e1;\n") -> Eliminated
{
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      + std::string(declare)
      + equations + "\n"
      "x1.lo = -10; x1.up = 10;\n"
      "Solve m using NLP " + sense + " objvar;\n");

  bool kept = false;
  for (auto const& name : parsed.var_names) {
    kept = kept || name == "objvar";
  }

  std::vector<double> point(parsed.problem.box_bounds.size(), 3.0);
  return {!kept,
          evaluate_objective(parsed.problem, point),
          parsed.problem.box_bounds.size()};
}

}  // namespace

TEST_CASE("objective rearrangement handles the shapes Convert emits",
          "[gams][elimination]")
{
  SECTION("canonical -(P) + objvar =E= 0")
  {
    auto r = try_eliminate("e1.. -(sqr(x1) + 2*x1) + objvar =E= 0;");
    CHECK(r.eliminated);
    CHECK(r.vars == 1);
    CHECK(close(r.value, 9.0 + 6.0));
  }

  SECTION("objvar - P =E= 0")
  {
    auto r = try_eliminate("e1.. objvar - sqr(x1) =E= 0;");
    CHECK(r.eliminated);
    CHECK(close(r.value, 9.0));
  }

  SECTION("P + objvar =E= constant, i.e. a non-zero right-hand side")
  {
    // ex8_6_2's own shape: `... + objvar =E= -45`.
    auto r = try_eliminate("e1.. -(sqr(x1)) + objvar =E= -45;");
    CHECK(r.eliminated);
    CHECK(close(r.value, -45.0 + 9.0));
  }

  SECTION("coefficient other than 1")
  {
    auto r = try_eliminate("e1.. 2*objvar - sqr(x1) =E= 0;");
    CHECK(r.eliminated);
    CHECK(close(r.value, 4.5));
  }

  SECTION("objvar on the right-hand side")
  {
    auto r = try_eliminate("e1.. sqr(x1) =E= objvar;");
    CHECK(r.eliminated);
    CHECK(close(r.value, 9.0));
  }

  SECTION("objvar divided by a constant")
  {
    auto r = try_eliminate("e1.. objvar/4 - sqr(x1) =E= 0;");
    CHECK(r.eliminated);
    CHECK(close(r.value, 36.0));
  }
}

TEST_CASE("nonlinear occurrences of the objective variable fall back cleanly",
          "[gams][elimination]")
{
  // Each of these must keep objvar as a variable rather than throw: the
  // fallback is what stops an unusual instance from being a hard failure.
  SECTION("objvar under a function")
  {
    auto r = try_eliminate("e1.. sqr(objvar) - x1 =E= 0;");
    CHECK_FALSE(r.eliminated);
    CHECK(r.vars == 2);
  }

  SECTION("objvar multiplied by a variable")
  {
    auto r = try_eliminate("e1.. objvar*x1 - 1 =E= 0;");
    CHECK_FALSE(r.eliminated);
  }

  SECTION("objvar in a denominator")
  {
    auto r = try_eliminate("e1.. x1/objvar - 1 =E= 0;");
    CHECK_FALSE(r.eliminated);
  }

  SECTION("objvar cancels out")
  {
    auto r = try_eliminate("e1.. objvar - objvar + x1 =E= 0;");
    CHECK_FALSE(r.eliminated);
  }

  SECTION("only an inequality mentions objvar, pointing the wrong way")
  {
    // objvar <= sqr(x1) under `minimizing` is unbounded below, not min sqr(x1).
    auto r = try_eliminate("e1.. objvar - sqr(x1) =L= 0;");
    CHECK_FALSE(r.eliminated);
  }
}

TEST_CASE("an inequality that is tight at the optimum eliminates objvar",
          "[gams][elimination]")
{
  // `min objvar s.t. objvar >= f(x)` is `min f(x)`: nothing pushes objvar
  // below f, so it sits on it at every optimum. The same rearrangement as the
  // =E= path, admissible only when the direction matches the sense.
  SECTION("autocorr_bern20-03's shape: f(x) - objvar =L= 0, minimising")
  {
    auto r = try_eliminate_ineq("e1.. sqr(x1) - objvar =L= 0;");
    CHECK(r.eliminated);
    CHECK(r.vars == 1);
    CHECK(close(r.value, 9.0));
  }

  SECTION("the same relation written =G=")
  {
    auto r = try_eliminate_ineq("e1.. -(sqr(x1)) + objvar =G= 0;");
    CHECK(r.eliminated);
    CHECK(close(r.value, 9.0));
  }

  SECTION("coefficient other than 1, and a non-zero right-hand side")
  {
    // 2*objvar >= sqr(x1) - 8, so objvar >= (sqr(x1) - 8)/2.
    auto r = try_eliminate_ineq("e1.. 2*objvar - sqr(x1) =G= -8;");
    CHECK(r.eliminated);
    CHECK(close(r.value, 0.5));
  }

  SECTION("maximising is the mirror image")
  {
    // max objvar s.t. objvar <= sqr(x1) is max sqr(x1); the reported objective
    // is negated because the driver only minimises.
    auto r = try_eliminate_ineq("e1.. sqr(x1) - objvar =G= 0;", "maximizing");
    CHECK(r.eliminated);
    CHECK(close(r.value, -9.0));
  }

  SECTION("the wrong direction for the sense falls back")
  {
    // objvar >= sqr(x1) under `maximizing` is unbounded above.
    auto r = try_eliminate_ineq("e1.. sqr(x1) - objvar =L= 0;", "maximizing");
    CHECK_FALSE(r.eliminated);
    CHECK(r.vars == 2);
  }

  SECTION("objvar in a second equation falls back")
  {
    // "Nothing else pushes objvar down" stops being true: e2 could make the
    // tight value infeasible, so the substitution is not sound.
    auto r = try_eliminate_ineq(
        "e1.. sqr(x1) - objvar =L= 0;\n" "e2.. objvar + x1 =L= 4;",
        "minimizing",
        "Equations e1,e2;\n");
    CHECK_FALSE(r.eliminated);
    CHECK(r.vars == 2);
  }

  SECTION("an =E= definition is preferred when the model has both")
  {
    // e2 would also be admissible; the equality is unconditionally valid, so a
    // model carrying both never relies on the weaker argument.
    auto r = try_eliminate_ineq(
        "e1.. -(sqr(x1)) + objvar =E= 0;\n" "e2.. 2*x1 - objvar =L= 0;",
        "minimizing",
        "Equations e1,e2;\n");
    CHECK(r.eliminated);
    CHECK(close(r.value, 9.0));
    // e2 is not the defining equation, so it stays a real constraint.
    CHECK(r.vars == 1);
  }
}

TEST_CASE(
    "a bound on the eliminated objective variable survives as a constraint",
    "[gams][elimination]")
{
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      "Equations e1;\n"
      "e1.. -(sqr(x1)) + objvar =E= 0;\n"
      "x1.lo = -10; x1.up = 10;\n"
      "objvar.up = 25;\n"
      "Solve m using NLP minimizing objvar;\n");

  REQUIRE(parsed.var_names.size() == 1);  // objvar is gone
  REQUIRE(parsed.problem.constraints.size() == 1);  // but its bound is not
  CHECK(parsed.problem.constraints[0].cmp == cuminlp::model::Cmp::LE);
  CHECK(close(parsed.problem.constraints[0].rhs, 25.0));
}

TEST_CASE("a surviving objvar bound is stated in the file's own sense",
          "[gams][elimination]")
{
  // The negation that turns `maximizing` into the driver's minimise is a
  // solver-facing rewrite; a bound the file states is about the value the file
  // wrote. `objvar.up = 25` while maximising means f(x) <= 25, so the
  // constraint must be aimed at f, not at -f -- which would silently mean
  // f(x) >= -25, a different feasible set.
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      "Equations e1;\n"
      "e1.. -(sqr(x1)) + objvar =E= 0;\n"
      "x1.lo = -10; x1.up = 10;\n"
      "objvar.up = 25;\n"
      "Solve m using NLP maximizing objvar;\n");

  REQUIRE(parsed.problem.constraints.size() == 1);
  CHECK(parsed.problem.constraints[0].cmp == cuminlp::model::Cmp::LE);
  CHECK(close(parsed.problem.constraints[0].rhs, 25.0));
  // f(3) = 9, not -9.
  CHECK(close(
      cuminlp::testing::evaluate(
          parsed.problem.graph, parsed.problem.constraints[0].root_id, {3.0}),
      9.0));
}

TEST_CASE("an inequality elimination keeps only the binding objvar bound",
          "[gams][elimination]")
{
  auto parse_bounded = [](char const* bounds, char const* sense)
  {
    return gams::parse<double>(
        "Variables x1,objvar;\n"
        "Equations e1;\n"
        "e1.. sqr(x1) - objvar =L= 0;\n"
        "x1.lo = -10; x1.up = 10;\n"
        + std::string(bounds)
        + "Solve m using NLP " + sense + " objvar;\n");
  };

  SECTION("an upper bound still restricts x and survives")
  {
    // No feasible objvar exists at all unless sqr(x1) <= 25.
    auto parsed = parse_bounded("objvar.up = 25;\n", "minimizing");
    REQUIRE(parsed.problem.constraints.size() == 1);
    CHECK(parsed.problem.constraints[0].cmp == cuminlp::model::Cmp::LE);
    CHECK(close(parsed.problem.constraints[0].rhs, 25.0));
  }

  SECTION("a lower bound restricts nothing and is dropped, loudly")
  {
    // Every x admits a feasible objvar (just a higher one), so `objvar.lo = 5`
    // is not a constraint on the model. Emitting it as sqr(x1) >= 5 would cut
    // off x1 = 0, which is feasible with objvar = 5.
    auto parsed = parse_bounded("objvar.lo = 5;\n", "minimizing");
    CHECK(parsed.problem.constraints.empty());

    bool mentioned = false;
    for (auto const& w : parsed.warnings) {
      mentioned = mentioned || w.message.find("objvar.lo") != std::string::npos;
    }
    CHECK(mentioned);
  }

  SECTION("the =E= path keeps both, since objvar is pinned there")
  {
    auto parsed = gams::parse<double>(
        "Variables x1,objvar;\n"
        "Equations e1;\n"
        "e1.. -(sqr(x1)) + objvar =E= 0;\n"
        "x1.lo = -10; x1.up = 10;\n"
        "objvar.lo = 5; objvar.up = 25;\n"
        "Solve m using NLP minimizing objvar;\n");
    CHECK(parsed.problem.constraints.size() == 2);
  }
}

TEST_CASE("maximise negates the objective and reports the original sense",
          "[gams][elimination]")
{
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      "Equations e1;\n"
      "e1.. -(sqr(x1)) + objvar =E= 0;\n"
      "x1.lo = -10; x1.up = 10;\n"
      "Solve m using NLP maximizing objvar;\n");
  CHECK(parsed.sense == gams::Sense::Maximise);
  CHECK(close(evaluate_objective(parsed.problem, {3.0}), -9.0));
}

// ---------------------------------------------------------------------------
// Layer 4: agreement with the hand-built DAGs already in the repo
// ---------------------------------------------------------------------------

namespace
{

/// x^n as an Op::PowN node. Written out rather than via an Expr helper so this
/// test depends on nothing beyond the committed dag.hpp.
auto pown(Problem<double>& problem, Expr<double> base, int n) -> Expr<double>
{
  cuminlp::model::DAGNodePayload<double> payload;
  payload.int_exp = n;
  return Expr<double>(
      &problem.graph,
      problem.graph.emit(cuminlp::model::Op::PowN, {base.id()}, payload));
}

/// Rebuild ex4_1_2 exactly as source/power_series.cu does.
auto hand_built_ex4_1_2() -> Problem<double>
{
  static constexpr double kCoeffs[] = {
      1.666666666, 1.25,        1.0,        0.8333333,    0.714285714,
      0.625,       0.555555555, 1.0,        -43.6363636,  0.41666666,
      0.384615384, 0.357142857, 0.3333333,  0.3125,       0.294117647,
      0.277777777, 0.263157894, 0.25,       0.238095238,  0.227272727,
      0.217391304, 0.208333333, 0.2,        0.192307692,  0.185185185,
      0.178571428, 0.344827586, 0.6666666,  -15.48387097, 0.15625,
      0.1515151,   0.14705882,  0.14285712, 0.138888888,  0.135135135,
      0.131578947, 0.128205128, 0.125,      0.121951219,  0.119047619,
      0.116279069, 0.113636363, 0.1111111,  0.108695652,  0.106382978,
      0.208333333, 0.408163265, 0.8};
  constexpr int kFirstPower = 3;

  Problem<double> problem;
  auto x = problem.var(1, 2);
  Expr<double> obj = (2.5 * (x * x)) - (500.0 * x);
  for (std::size_t i = 0; i < std::size(kCoeffs); ++i) {
    obj = obj
        + (kCoeffs[i] * pown(problem, x, kFirstPower + static_cast<int>(i)));
  }
  problem.set_objective(obj);
  return problem;
}

/// Rebuild ex8_6_2 exactly as source/morse_cluster_energy.cu does.
auto hand_built_ex8_6_2() -> Problem<double>
{
  constexpr std::size_t kPoints = 10;
  constexpr std::size_t kVars = 3 * kPoints;

  Problem<double> problem;
  std::vector<Expr<double>> x;
  x.reserve(kVars);
  for (std::size_t i = 0; i < kVars; ++i) {
    if (i == 0 || i == 10 || i == 11 || i == 20 || i == 21 || i == 22) {
      x.push_back(problem.fixed(0));
    } else {
      x.push_back(problem.var(-5, 5));
    }
  }

  std::vector<Expr<double>> terms;
  for (std::size_t i = 0; i < kPoints; ++i) {
    for (std::size_t j = i + 1; j < kPoints; ++j) {
      auto dx = x[i] - x[j];
      auto dy = x[i + 10] - x[j + 10];
      auto dz = x[i + 20] - x[j + 20];
      auto r = sqrt((dx * dx) + (dy * dy) + (dz * dz));
      auto u = 1 - exp(3 * (1 - r));
      terms.push_back(u * u);
    }
  }

  Expr<double> obj = terms[0];
  for (std::size_t i = 1; i < terms.size(); ++i) {
    obj = obj + terms[i];
  }
  problem.set_objective(-45 + obj);
  return problem;
}

/// Evaluate both DAGs at the same pseudo-random points inside the shared box.
void check_agrees(Problem<double> const& reference,
                  gams::ParsedModel<double> const& parsed,
                  int samples,
                  double tol)
{
  REQUIRE(parsed.problem.box_bounds.size() == reference.box_bounds.size());

  std::mt19937 rng(12345);
  for (int s = 0; s < samples; ++s) {
    std::vector<double> point;
    point.reserve(reference.box_bounds.size());
    for (auto const& b : reference.box_bounds) {
      std::uniform_real_distribution<double> pick(b.lb, b.ub);
      point.push_back(pick(rng));
    }
    double const want = evaluate_objective(reference, point);
    double const got = evaluate_objective(parsed.problem, point);
    INFO("sample " << s << " expected " << want << " got " << got);
    REQUIRE(close(got, want, tol));
  }
}

}  // namespace

TEST_CASE("ex4_1_2.gms agrees with the hand-built DAG", "[gams][minlplib]")
{
  auto parsed = gams::parse_file<double>(data_file("ex4_1_2.gms"));
  auto reference = hand_built_ex4_1_2();

  CHECK(parsed.var_names.size() == 1);
  CHECK(parsed.var_names[0] == "x1");
  CHECK(parsed.problem.constraints.empty());
  CHECK(close(parsed.problem.box_bounds[0].lb, 1.0));
  CHECK(close(parsed.problem.box_bounds[0].ub, 2.0));

  // Degree-50 polynomial on [1, 2] reaches ~1e9, so compare relatively.
  check_agrees(reference, parsed, 64, 1e-12);
}

TEST_CASE("ex8_6_2.gms agrees with the hand-built DAG", "[gams][minlplib]")
{
  auto parsed = gams::parse_file<double>(data_file("ex8_6_2.gms"));
  auto reference = hand_built_ex8_6_2();

  // 30 coordinates, 6 fixed to 0 to break translational/rotational symmetry.
  CHECK(parsed.var_names.size() == 24);
  CHECK(parsed.var_names.front() == "x2");
  CHECK(parsed.var_names.back() == "x30");
  CHECK(parsed.problem.constraints.empty());

  check_agrees(reference, parsed, 64, 1e-12);
}

TEST_CASE("fixed variables become constants, not degenerate boxes",
          "[gams][minlplib]")
{
  auto parsed = gams::parse_file<double>(data_file("ex8_6_2.gms"));
  for (auto const& b : parsed.problem.box_bounds) {
    CHECK(b.lb < b.ub);  // no [v, v] dimensions survived
  }

  gams::ParseOptions keep_them;
  keep_them.fold_fixed_to_const = false;
  auto unfolded = gams::parse_file<double>(data_file("ex8_6_2.gms"), keep_them);
  CHECK(unfolded.var_names.size() == 30);
}

// ---------------------------------------------------------------------------
// Layer 4b: integer and binary variables
// ---------------------------------------------------------------------------

TEST_CASE("integer variables parse as VarKind::Integer",
          "[gams][minlplib][discrete]")
{
  // nvs01.gms declares i1,i2 as Integer (implicit lower bound 0) and narrows
  // their upper bound with `i1.up = 200;` / `i2.up = 200;`; x3 stays
  // continuous. This is the file the frontend used to reject outright.
  auto parsed = gams::parse_file<double>(data_file("nvs01.gms"));

  REQUIRE(parsed.problem.var_kinds.size() == 3);
  CHECK(parsed.problem.var_kinds[0] == cuminlp::model::VarKind::Integer);
  CHECK(parsed.problem.var_kinds[1] == cuminlp::model::VarKind::Integer);
  CHECK(parsed.problem.var_kinds[2] == cuminlp::model::VarKind::Continuous);

  CHECK(close(parsed.problem.box_bounds[0].lb, 0.0));
  CHECK(close(parsed.problem.box_bounds[0].ub, 200.0));
  CHECK(close(parsed.problem.box_bounds[1].lb, 0.0));
  CHECK(close(parsed.problem.box_bounds[1].ub, 200.0));
}

TEST_CASE("Problem::validate() defects the frontend must not reproduce",
          "[gams][discrete][validate]")
{
  SECTION("fractional bounds on an integer snap inward")
  {
    auto parsed = gams::parse<double>(
        "Variables i1,objvar;\n"
        "Integer Variables i1;\n"
        "Equations e1;\n"
        "e1.. -(i1) + objvar =E= 0;\n"
        "i1.lo = 2.3; i1.up = 7.8;\n"
        "Solve m using MINLP minimizing objvar;\n");
    CHECK(close(parsed.problem.box_bounds[0].lb, 3.0));  // ceil(2.3)
    CHECK(close(parsed.problem.box_bounds[0].ub, 7.0));  // floor(7.8)
  }

  SECTION("an integer-empty box is rejected, not silently truncated")
  {
    // [0.3, 0.7] contains no integer at all, but isn't lo > up before
    // snapping -- the empty-box check must run after ceil/floor, not before.
    CHECK_THROWS_AS(gams::parse<
                        double>("Variables i1,objvar;\n" "Integer Variables "
                                                         "i1;\n" "Equations "
                                                                 "e1;\n" "e1.. "
                                                                         "-(i1)"
                                                                         " + "
                                                                         "objva"
                                                                         "r "
                                                                         "=E= "
                                                                         "0;\n" "i1.lo = 0.3; i1.up = 0.7;\n" "Solve m using MINLP minimizing objvar;\n"),
                    gams::ParseError);
  }

  SECTION("a binary narrowed off [0,1] lowers to VarKind::Integer")
  {
    // .up/.lo narrowing is not fixing, so this survives even with
    // fold_fixed_to_const on, and Problem::validate() requires Binary boxes
    // to be exactly [0,1].
    auto parsed = gams::parse<double>(
        "Variables b1,objvar;\n"
        "Binary Variables b1;\n"
        "Equations e1;\n"
        "e1.. -(b1) + objvar =E= 0;\n"
        "b1.up = 0;\n"
        "Solve m using MINLP minimizing objvar;\n");
    CHECK(parsed.problem.var_kinds[0] == cuminlp::model::VarKind::Integer);
    CHECK(close(parsed.problem.box_bounds[0].lb, 0.0));
    CHECK(close(parsed.problem.box_bounds[0].ub, 0.0));
  }

  SECTION("integrality is sticky across a narrowing re-declaration")
  {
    // GAMS Convert always emits `Variables` first and domain-narrowing
    // declarations after, so this ordering doesn't occur in the corpus -- but
    // nothing stops a hand-written file from doing it, and it must not
    // silently clear i1's integrality.
    auto parsed = gams::parse<double>(
        "Integer Variables i1;\n"
        "Positive Variables i1;\n"
        "Variables objvar;\n"
        "Equations e1;\n"
        "e1.. -(i1) + objvar =E= 0;\n"
        "i1.up = 10;\n"
        "Solve m using MINLP minimizing objvar;\n");
    CHECK(parsed.problem.var_kinds[0] == cuminlp::model::VarKind::Integer);
  }
}

TEST_CASE("validate() passes on every fixture in test/data/gams/",
          "[gams][minlplib][validate]")
{
  // Would have caught every Problem::validate() defect above had it existed
  // before they were fixed: parse_file() calls validate() internally and
  // turns any throw into a ParseError with ErrorKind::Unrepresentable, so
  // that specific kind is what must never come out of this loop. Some
  // fixtures may still legitimately fail for an unrelated, pre-existing
  // reason (e.g. a function this frontend doesn't lower yet) -- that's a
  // frontier gap, not a validate() defect, and isn't what this test checks.
  for (auto const& entry :
       std::filesystem::directory_iterator(CUMINLP_GAMS_TEST_DATA))
  {
    if (entry.path().extension() != ".gms") {
      continue;
    }
    INFO("file: " << entry.path());
    try {
      gams::parse_file<double>(entry.path());
    } catch (gams::ParseError const& e) {
      CHECK(e.kind != gams::ErrorKind::Unrepresentable);
    }
  }
}

TEST_CASE("nvs04.gms agrees with the hand-built DAG",
          "[gams][minlplib][discrete]")
{
  // Rebuild nvs04 (https://www.minlplib.org/nvs04.html) by hand: a
  // Rosenbrock-shaped objective over two integers in [0, 200]. Its defining
  // equation is `=E=`, so it eliminates cleanly: 2 integers, no leftover
  // objvar/constraint. (nvs09, this project's original choice, turned out to
  // need `x**0.2` -- a real, non-0.5 exponent, which this frontend now lowers
  // to exp(0.2*log(x)) rather than rejecting -- see "x**c lowers to
  // exp(c*log(x))..." below -- so nvs04 is the equivalence fixture instead;
  // nvs09_problem.hpp stays as source/nvs09.cu's hand-built version, itself
  // written as exp(0.2*log(product)) for exactly this reason.)
  Problem<double> reference;
  auto i1 = reference.int_var(0, 200);
  auto i2 = reference.int_var(0, 200);
  reference.set_objective(100.0 * sqr(0.5 - sqr(0.6 + i1) + i2)
                          + sqr(0.4 - i1));

  auto parsed = gams::parse_file<double>(data_file("nvs04.gms"));

  REQUIRE(parsed.problem.var_kinds.size() == 2);
  CHECK(parsed.problem.var_kinds[0] == cuminlp::model::VarKind::Integer);
  CHECK(parsed.problem.var_kinds[1] == cuminlp::model::VarKind::Integer);
  CHECK(parsed.problem.constraints.empty());

  std::mt19937 rng(20260730);
  std::uniform_int_distribution<int> pick(0,
                                          200);  // lattice points in [0, 200]
  for (int s = 0; s < 32; ++s) {
    std::vector<double> point = {static_cast<double>(pick(rng)),
                                 static_cast<double>(pick(rng))};
    double const want = evaluate_objective(reference, point);
    double const got = evaluate_objective(parsed.problem, point);
    INFO("sample " << s << " expected " << want << " got " << got);
    CHECK(close(got, want, 1e-9));
  }
}

TEST_CASE("autocorr_bern20-03.gms agrees with the hand-built DAG",
          "[gams][minlplib][discrete]")
{
  using cuminlp::examples::autocorr_bern20_03::make_autocorr_bern20_03;

  // Its defining equation is `=L=`, not `=E=`, and `f(b) - objvar =L= 0` under
  // `minimizing` is tight at every optimum -- so objvar is eliminated through
  // it and this parses to exactly make_autocorr_bern20_03()'s shape: 20
  // binaries, no leftover objvar dimension, no constraint. It did not always:
  // while the inequality path was unimplemented, the fallback kept objvar as a
  // 21st continuous variable with a +-1e6 default box, which pinned the search
  // driver's dual bound at -1e6 (the objective being the bare variable) and
  // left the instance unsolvable at any search shape.
  auto parsed = gams::parse_file<double>(data_file("autocorr_bern20-03.gms"));
  auto reference = make_autocorr_bern20_03();

  REQUIRE(parsed.problem.var_kinds.size() == 20);
  for (std::size_t i = 0; i < 20; ++i) {
    CHECK(parsed.problem.var_kinds[i] == cuminlp::model::VarKind::Binary);
  }
  CHECK(parsed.problem.constraints.empty());

  std::mt19937 rng(20260730);
  std::uniform_int_distribution<int> pick(0, 1);
  for (int s = 0; s < 32; ++s) {
    std::vector<double> b;
    b.reserve(20);
    for (int j = 0; j < 20; ++j) {
      b.push_back(static_cast<double>(pick(rng)));
    }

    double const want = evaluate_objective(reference, b);
    double const got = evaluate_objective(parsed.problem, b);
    INFO("sample " << s << " expected " << want << " got " << got);
    CHECK(close(got, want, 1e-9));
  }
}

// ---------------------------------------------------------------------------
// Layer 5: diagnostics
// ---------------------------------------------------------------------------

TEST_CASE("unrepresentable input is rejected with a line number",
          "[gams][errors]")
{
  auto fails_at = [](std::string const& source, int line)
  {
    try {
      gams::parse<double>(source);
    } catch (gams::ParseError const& e) {
      CHECK(e.line == line);
      return true;
    }
    return false;
  };

  SECTION("unsupported function")
  {
    CHECK(fails_at(model_with("arctan(x1)"), 3));
  }

  SECTION("unknown function")
  {
    CHECK(fails_at(model_with("wibble(x1)"), 3));
  }

  SECTION("non-scalar format")
  {
    CHECK(fails_at("Set i /1*10/;\n", 1));
  }

  SECTION("undeclared identifier")
  {
    CHECK(fails_at(model_with("zzz"), 3));
  }

  SECTION("no Solve statement")
  {
    CHECK_THROWS_AS(gams::parse<double>("Variables x1;\n"), gams::ParseError);
  }
}

TEST_CASE("free variables get a default box and say so", "[gams][errors]")
{
  gams::ParseOptions options;
  options.default_bound = 500.0;
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      "Equations e1;\n"
      "e1.. -(x1) + objvar =E= 0;\n"
      "Solve m using NLP minimizing objvar;\n",
      options);

  CHECK(close(parsed.problem.box_bounds[0].lb, -500.0));
  CHECK(close(parsed.problem.box_bounds[0].ub, 500.0));
  CHECK(parsed.warnings.size() == 2);
}

TEST_CASE("relations canonicalise into LE and EQ", "[gams][constraints]")
{
  auto parsed = gams::parse<double>(
      "Variables x1,objvar;\n"
      "Equations e1,e2,e3;\n"
      "e1.. -(x1) + objvar =E= 0;\n"
      "e2.. 2*x1 =L= 8;\n"
      "e3.. 3*x1 =G= 6;\n"
      "x1.lo = 0; x1.up = 10;\n"
      "Solve m using NLP minimizing objvar;\n");

  REQUIRE(parsed.problem.constraints.size() == 2);
  auto const& graph = parsed.problem.graph;

  // e2 keeps its constant right-hand side out of the DAG.
  CHECK(parsed.problem.constraints[0].cmp == cuminlp::model::Cmp::LE);
  CHECK(close(parsed.problem.constraints[0].rhs, 8.0));

  // e3 `3*x1 =G= 6` is stored by swapping sides into `6 =L= 3*x1`, which has a
  // non-constant right-hand side, so it lowers to `6 - 3*x1 <= 0`.
  CHECK(parsed.problem.constraints[1].cmp == cuminlp::model::Cmp::LE);
  CHECK(close(parsed.problem.constraints[1].rhs, 0.0));
  double const body = cuminlp::testing::evaluate(
      graph, parsed.problem.constraints[1].root_id, {5.0});
  CHECK(close(body, 6.0 - 15.0));
}
