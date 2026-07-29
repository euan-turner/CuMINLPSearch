#include <cmath>
#include <limits>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>
#include <cuinterval/interval.h>

#include "cuminlp/dag.hpp"
#include "cuminlp/errors.hpp"

using cuminlp::InvalidDAG;
using cuminlp::InvalidProblem;
using cuminlp::dag::Op;
using cuminlp::dag::Problem;
using cuminlp::dag::VarKind;

namespace {
// The constants under test here are all either passed straight through
// (e.g. bin_var()'s bounds) or computed by a single well-defined libm call
// on an exactly-representable input (e.g. std::sqrt(4.0)), so bitwise
// equality is the right check -- this sidesteps -Wfloat-equal (which flags
// == / != between floats, but not < / >) rather than weakening the
// assertion to an approximate one.
bool feq(double a, double b) { return !(a < b) && !(b < a); }
}  // namespace

// TEST_EXTENSION.md §1a: Problem is move-only, not copyable -- a copy would
// leave every previously-issued Expr aliasing the original's graph. This is
// a structural (compile-time) invariant, not a runtime one.
static_assert(!std::is_copy_constructible_v<Problem<double>>);
static_assert(!std::is_copy_assignable_v<Problem<double>>);
static_assert(std::is_move_constructible_v<Problem<double>>);
static_assert(std::is_move_assignable_v<Problem<double>>);

TEST_CASE("Problem::var/int_var/bin_var append exactly one entry each, in order", "[dag]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  auto y = p.int_var(0.0, 5.0);
  auto z = p.bin_var();

  REQUIRE(p.box_bounds.size() == 3);
  REQUIRE(p.var_kinds.size() == 3);

  CHECK(p.var_kinds[0] == VarKind::Continuous);
  CHECK(p.var_kinds[1] == VarKind::Integer);
  CHECK(p.var_kinds[2] == VarKind::Binary);

  // Op::Var payload.var_index matches position in box_bounds/var_kinds.
  CHECK(p.graph.nodes[x.id()].payload.var_index == 0);
  CHECK(p.graph.nodes[y.id()].payload.var_index == 1);
  CHECK(p.graph.nodes[z.id()].payload.var_index == 2);

  CHECK(feq(p.box_bounds[2].lb, 0.0));
  CHECK(feq(p.box_bounds[2].ub, 1.0));
}

TEST_CASE("Problem::fixed never registers as a variable", "[dag]")
{
  Problem<double> p;
  p.var(-1.0, 1.0);
  auto before = p.box_bounds.size();

  p.fixed(5.0);

  CHECK(p.box_bounds.size() == before);
  CHECK(p.var_kinds.size() == before);
}

TEST_CASE("DAGNode ids are topologically ordered and op_count reflects subtree size", "[dag]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  auto y = p.var(0.0, 5.0);
  auto e = x * x + 2.0 * y;  // Sqr(x) [op_count 2], 2.0*y [Const, Mul -> op_count 3], Add -> 6
  p.set_objective(e);
  p.add_constraint(x + y, cuminlp::dag::Cmp::LE, 5.0);

  for (const auto& node : p.graph.nodes) {
    for (auto in_id : node.in) {
      CHECK(in_id < node.id);
    }
  }

  REQUIRE_NOTHROW(p.validate());
}

TEST_CASE("Reusing an Expr handle in two places does not corrupt op_count bookkeeping", "[dag]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  auto shared = x * x;  // Sqr(x), op_count == 2

  auto objective = shared + shared;   // reused in the objective ...
  p.set_objective(objective);
  p.add_constraint(shared, cuminlp::dag::Cmp::LE, 1.0);  // ... and a constraint

  CHECK(p.graph.nodes[shared.id()].op_count == 2);
  // objective = Add(shared, shared): 1 + 2 + 2 = 5
  CHECK(p.graph.nodes[objective.id()].op_count == 5);
  REQUIRE_NOTHROW(p.validate());
}

TEST_CASE("a*a only becomes Op::Sqr when both operands are the same node", "[dag]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  auto y = p.var(-1.0, 1.0);

  auto self_product = x * x;
  auto cross_product = x * y;

  CHECK(p.graph.nodes[self_product.id()].op == Op::Sqr);
  CHECK(p.graph.nodes[cross_product.id()].op == Op::Mul);
}

TEST_CASE("live_set_estimate is always 0", "[dag]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  auto e = sqrt(x * x);
  CHECK(p.graph.nodes[x.id()].live_set_estimate == 0);
  CHECK(p.graph.nodes[e.id()].live_set_estimate == 0);
}

// §1b: a unary op applied directly to a Const constant-folds rather than
// emitting a degenerate op node over a Const.
TEST_CASE("Unary ops applied directly to fixed() constant-fold", "[dag][1b]")
{
  Problem<double> p;

  auto neg = -p.fixed(2.0);
  CHECK(p.graph.nodes[neg.id()].op == Op::Const);
  CHECK(feq(p.graph.nodes[neg.id()].payload.constant, -2.0));

  auto sqrt_c = sqrt(p.fixed(4.0));
  CHECK(p.graph.nodes[sqrt_c.id()].op == Op::Const);
  CHECK(feq(p.graph.nodes[sqrt_c.id()].payload.constant, 2.0));

  auto exp_c = exp(p.fixed(0.0));
  CHECK(p.graph.nodes[exp_c.id()].op == Op::Const);
  CHECK(feq(p.graph.nodes[exp_c.id()].payload.constant, 1.0));

  auto log_c = log(p.fixed(1.0));
  CHECK(p.graph.nodes[log_c.id()].op == Op::Const);
  CHECK(feq(p.graph.nodes[log_c.id()].payload.constant, 0.0));

  auto pow_c = pow(p.fixed(2.0), 3);
  CHECK(p.graph.nodes[pow_c.id()].op == Op::Const);
  CHECK(feq(p.graph.nodes[pow_c.id()].payload.constant, 8.0));

  auto sqr_c = sqr(p.fixed(3.0));
  CHECK(p.graph.nodes[sqr_c.id()].op == Op::Const);
  CHECK(feq(p.graph.nodes[sqr_c.id()].payload.constant, 9.0));
}

TEST_CASE("A bare Const as the objective root is rejected by validate()", "[dag][1b]")
{
  Problem<double> p;
  p.var(-1.0, 1.0);
  p.set_objective(p.fixed(5.0));

  REQUIRE_THROWS_AS(p.validate(), InvalidDAG);
}

TEST_CASE("A bare Const as a constraint root is rejected by validate()", "[dag][1b]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  p.set_objective(x * x);
  p.add_constraint(p.fixed(1.0), cuminlp::dag::Cmp::LE, 0.0);

  REQUIRE_THROWS_AS(p.validate(), InvalidDAG);
}

// §2: Problem construction invariants, checked via validate().

TEST_CASE("validate() accepts a well-formed Problem", "[dag][2]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  auto y = p.int_var(0.0, 5.0);
  p.set_objective(x * x + y);
  p.add_constraint(x + y, cuminlp::dag::Cmp::LE, 5.0);

  REQUIRE_NOTHROW(p.validate());
}

TEST_CASE("validate() rejects lb > ub", "[dag][2]")
{
  Problem<double> p;
  auto x = p.var(1.0, -1.0);  // lb > ub
  p.set_objective(x * x);

  REQUIRE_THROWS_AS(p.validate(), InvalidProblem);
}

TEST_CASE("validate() rejects non-integer bounds on an Integer variable", "[dag][2]")
{
  Problem<double> p;
  auto x = p.var(0.5, 5.0, VarKind::Integer);
  p.set_objective(x * x);

  REQUIRE_THROWS_AS(p.validate(), InvalidProblem);
}

TEST_CASE("validate() rejects a Binary variable whose bounds aren't exactly [0,1]", "[dag][2]")
{
  Problem<double> p;
  // Bypasses bin_var() (which always produces [0,1]) to exercise the
  // general var(lb, ub, kind) overload directly.
  auto x = p.var(0.0, 2.0, VarKind::Binary);
  p.set_objective(x * x);

  REQUIRE_THROWS_AS(p.validate(), InvalidProblem);
}

TEST_CASE("bin_var() always produces bounds accepted by validate()", "[dag][2]")
{
  Problem<double> p;
  auto x = p.bin_var();
  p.set_objective(x * x);

  REQUIRE_NOTHROW(p.validate());
}

TEST_CASE("validate() rejects a Problem whose objective was never set", "[dag][2]")
{
  Problem<double> p;
  p.var(-1.0, 1.0);

  REQUIRE_THROWS_AS(p.validate(), InvalidProblem);
}

TEST_CASE("validate() rejects a Problem with zero variables", "[dag][2]")
{
  Problem<double> p;
  p.set_objective(p.fixed(1.0));  // also hits the Const-root case, either error is fine here

  REQUIRE_THROWS(p.validate());
}

TEST_CASE("validate() rejects a non-finite constraint rhs", "[dag][2]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  p.set_objective(x * x);
  p.add_constraint(x, cuminlp::dag::Cmp::LE, std::numeric_limits<double>::infinity());

  REQUIRE_THROWS_AS(p.validate(), InvalidProblem);
}

TEST_CASE("validate() rejects a NaN constraint rhs", "[dag][2]")
{
  Problem<double> p;
  auto x = p.var(-1.0, 1.0);
  p.set_objective(x * x);
  p.add_constraint(x, cuminlp::dag::Cmp::EQ, std::numeric_limits<double>::quiet_NaN());

  REQUIRE_THROWS_AS(p.validate(), InvalidProblem);
}
