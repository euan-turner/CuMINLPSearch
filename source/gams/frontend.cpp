#include "cuminlp/gams.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ast.hpp"

namespace cuminlp::gams
{

using namespace cuminlp::gams::detail;

ParseError::ParseError(int line, std::string const& message, ErrorKind kind)
    : std::runtime_error("line " + std::to_string(line) + ": " + message)
    , line(line)
    , kind(kind)
{
}

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

enum class Tok {
  End, Ident, Number, Plus, Minus, Star, Slash, Pow,
  LParen, RParen, Comma, Semi, Assign, Rel, DotDot, Dot
};

struct Token {
  Tok kind = Tok::End;
  std::string text;  // identifiers, case-folded to lower
  double value = 0.0;
  Rel rel = Rel::E;
  int line = 1;
};

auto fold_case(std::string_view s) -> std::string
{
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

auto is_ident_start(char c) -> bool
{
  return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
}

auto is_ident_char(char c) -> bool
{
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

/**
 * Tokenise GAMS scalar source.
 *
 * Two rules are column-sensitive and must be applied before anything else:
 * `*` in column 1 is a line comment (elsewhere it is multiplication), and `$`
 * in column 1 is a dollar-control line (skipped wholesale, so its quotes and
 * macro syntax never reach the token stream).
 */
auto lex(std::string_view src) -> std::vector<Token>
{
  std::vector<Token> out;
  std::size_t i = 0;
  int line = 1;
  bool at_line_start = true;

  auto skip_line = [&] {
    while (i < src.size() && src[i] != '\n') ++i;
  };

  while (i < src.size()) {
    char c = src[i];

    if (c == '\n') {
      ++i;
      ++line;
      at_line_start = true;
      continue;
    }

    if (at_line_start && c == '*') {
      skip_line();
      continue;
    }

    if (at_line_start && c == '$') {
      std::size_t begin = i;
      skip_line();
      std::string directive = fold_case(src.substr(begin, i - begin));
      if (directive.rfind("$ontext", 0) == 0) {
        // Block comment: consume lines until one starts with $offtext.
        while (i < src.size()) {
          ++i;  // step over the newline
          ++line;
          std::size_t ls = i;
          skip_line();
          if (fold_case(src.substr(ls, i - ls)).rfind("$offtext", 0) == 0) break;
        }
      }
      continue;
    }

    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      // Any character, including a space, ends column 1. Continuation lines in
      // Convert output routinely start with indented `*` (`     *x2 + ...`),
      // which is multiplication, not a comment.
      at_line_start = false;
      ++i;
      continue;
    }

    at_line_start = false;
    Token t;
    t.line = line;

    if (is_ident_start(c)) {
      std::size_t begin = i;
      while (i < src.size() && is_ident_char(src[i])) ++i;
      t.kind = Tok::Ident;
      t.text = fold_case(src.substr(begin, i - begin));
      out.push_back(std::move(t));
      continue;
    }

    bool number_here = std::isdigit(static_cast<unsigned char>(c)) != 0
        || (c == '.' && i + 1 < src.size()
            && std::isdigit(static_cast<unsigned char>(src[i + 1])) != 0);
    if (number_here) {
      char const* begin = src.data() + i;
      char* end = nullptr;
      double v = std::strtod(begin, &end);
      if (end == begin) throw ParseError(line, "malformed number");
      i += static_cast<std::size_t>(end - begin);
      t.kind = Tok::Number;
      t.value = v;
      out.push_back(std::move(t));
      continue;
    }

    // `%NLP%` and friends: opaque to us, but must not break tokenising.
    if (c == '%') {
      std::size_t begin = i++;
      while (i < src.size() && src[i] != '%' && src[i] != '\n') ++i;
      if (i < src.size() && src[i] == '%') ++i;
      t.kind = Tok::Ident;
      t.text = fold_case(src.substr(begin, i - begin));
      out.push_back(std::move(t));
      continue;
    }

    switch (c) {
      case '+': t.kind = Tok::Plus; ++i; break;
      case '-': t.kind = Tok::Minus; ++i; break;
      case '/': t.kind = Tok::Slash; ++i; break;
      case '(': t.kind = Tok::LParen; ++i; break;
      case ')': t.kind = Tok::RParen; ++i; break;
      case ',': t.kind = Tok::Comma; ++i; break;
      case ';': t.kind = Tok::Semi; ++i; break;
      case '*':
        if (i + 1 < src.size() && src[i + 1] == '*') { t.kind = Tok::Pow; i += 2; }
        else { t.kind = Tok::Star; ++i; }
        break;
      case '.':
        if (i + 1 < src.size() && src[i + 1] == '.') { t.kind = Tok::DotDot; i += 2; }
        else { t.kind = Tok::Dot; ++i; }
        break;
      case '=':
        // `=E=` / `=L=` / `=G=` / `=N=`, versus a plain assignment.
        if (i + 2 < src.size() && std::isalpha(static_cast<unsigned char>(src[i + 1])) != 0
            && src[i + 2] == '=')
        {
          t.kind = Tok::Rel;
          switch (std::tolower(static_cast<unsigned char>(src[i + 1]))) {
            case 'e': t.rel = Rel::E; break;
            case 'l': t.rel = Rel::L; break;
            case 'g': t.rel = Rel::G; break;
            case 'n': t.rel = Rel::N; break;
            default: throw ParseError(line, "unknown relational operator");
          }
          i += 3;
        } else {
          t.kind = Tok::Assign;
          ++i;
        }
        break;
      case '\'':
      case '"':
        // A quoted set-element label, as in `s1s1('86')`. Quoted labels only
        // appear on indexed identifiers, which scalar format does not have.
        throw ParseError(line,
                         "quoted set element: this file has indexed identifiers "
                         "and is not in GAMS scalar format; regenerate it with "
                         "GAMS Convert",
                         ErrorKind::NonScalar);
      default:
        throw ParseError(line,
                         std::string("unexpected character '") + c + "'");
    }
    out.push_back(std::move(t));
  }

  Token end;
  end.kind = Tok::End;
  end.line = line;
  out.push_back(end);
  return out;
}

// ---------------------------------------------------------------------------
// Name tables
// ---------------------------------------------------------------------------

auto supported_functions() -> std::unordered_map<std::string, Func> const&
{
  static std::unordered_map<std::string, Func> const table = {
      {"sqr", Func::Sqr},     {"sqrt", Func::Sqrt},   {"exp", Func::Exp},
      {"log", Func::Log},     {"log10", Func::Log10}, {"log2", Func::Log2},
      {"abs", Func::Abs},     {"sin", Func::Sin},     {"cos", Func::Cos},
      {"tanh", Func::Tanh},   {"min", Func::Min},     {"max", Func::Max},
      {"power", Func::Power}, {"rpower", Func::Power},
  };
  return table;
}

/// Known GAMS functions we deliberately refuse rather than approximate.
/// Rewriting these (e.g. sinh via exp) is numerically correct but loses
/// interval tightness, which is exactly what this solver must not do silently.
auto unsupported_functions() -> std::unordered_set<std::string> const&
{
  static std::unordered_set<std::string> const table = {
      "tan", "arctan", "arcsin", "arccos", "errorf", "sign", "signpower",
      "ceil", "floor", "round", "trunc", "mod", "div", "centropy", "sinh",
      "cosh", "gamma", "loggamma", "beta", "entropy", "sigmoid", "slexp",
      "sqexp", "edist", "poly", "ifthen", "vcpower", "cvpower", "ncpcm",
  };
  return table;
}

/// Statement keywords that mean the file is not in scalar format.
auto non_scalar_keywords() -> std::unordered_set<std::string> const&
{
  static std::unordered_set<std::string> const table = {
      "set", "sets", "parameter", "parameters", "table", "alias", "scalar",
      "scalars", "loop", "sum", "prod", "smax", "smin", "abort", "display",
      "file", "put", "repeat", "while", "for",
  };
  return table;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

class Parser
{
public:
  Parser(std::vector<Token> tokens, Model& model, std::vector<Diagnostic>& warnings)
      : tokens_(std::move(tokens))
      , m_(model)
      , warnings_(warnings)
  {
  }

  void run()
  {
    while (cur().kind != Tok::End) statement();
  }

private:
  // -- token cursor ---------------------------------------------------------

  auto cur() const -> Token const& { return tokens_[pos_]; }
  auto peek(std::size_t n = 1) const -> Token const&
  {
    return tokens_[std::min(pos_ + n, tokens_.size() - 1)];
  }
  void advance() { if (pos_ + 1 < tokens_.size()) ++pos_; }

  auto is_word(std::string_view w) const -> bool
  {
    return cur().kind == Tok::Ident && cur().text == w;
  }

  void expect(Tok k, char const* what)
  {
    if (cur().kind != k) throw ParseError(cur().line, std::string("expected ") + what);
    advance();
  }

  void skip_to_semi()
  {
    while (cur().kind != Tok::Semi && cur().kind != Tok::End) advance();
    if (cur().kind == Tok::Semi) advance();
  }

  void warn(int line, std::string message)
  {
    warnings_.push_back({line, std::move(message)});
  }

  // -- statements -----------------------------------------------------------

  void statement()
  {
    if (cur().kind == Tok::Semi) { advance(); return; }
    if (cur().kind != Tok::Ident) {
      throw ParseError(cur().line, "expected the start of a statement");
    }

    std::string const& w = cur().text;

    if (non_scalar_keywords().count(w) != 0) {
      throw ParseError(cur().line,
                       "'" + w + "' means this file is not in GAMS scalar format; "
                       "regenerate it with GAMS Convert",
                       ErrorKind::NonScalar);
    }
    if (w == "sos1" || w == "sos2" || w == "semicont" || w == "semiint") {
      throw ParseError(cur().line, "'" + w + "' variables are not supported",
                       ErrorKind::Discrete);
    }

    if (w == "free" || w == "positive" || w == "negative" || w == "binary"
        || w == "integer")
    {
      declare_variables(w);
      return;
    }
    if (w == "variables" || w == "variable") { declare_variables("free"); return; }
    if (w == "equations" || w == "equation") { declare_equations(); return; }
    if (w == "model" || w == "models" || w == "option" || w == "options") {
      skip_to_semi();
      return;
    }
    if (w == "solve") { solve_statement(); return; }

    if (peek().kind == Tok::DotDot) { equation_definition(); return; }
    if (peek().kind == Tok::Dot) { attribute_assignment(); return; }

    throw ParseError(cur().line, "unrecognised statement starting at '" + w + "'");
  }

  void declare_variables(std::string const& domain)
  {
    int line = cur().line;
    advance();  // the domain word, or `variables` itself
    if (is_word("variables") || is_word("variable")) advance();

    double lo = -kInf;
    double up = kInf;
    dag::VarKind kind = dag::VarKind::Continuous;
    if (domain == "positive") { lo = 0.0; }
    else if (domain == "negative") { up = 0.0; }
    else if (domain == "binary") { lo = 0.0; up = 1.0; kind = dag::VarKind::Binary; }
    else if (domain == "integer") { lo = 0.0; kind = dag::VarKind::Integer; }

    while (true) {
      if (cur().kind != Tok::Ident) throw ParseError(cur().line, "variable name");
      // A repeat declaration (`Positive Variables x1;` after `Variables x1;`)
      // narrows the domain of an existing symbol; it does not create a new one,
      // so declaration order -- and therefore var_index -- comes from the first
      // mention.
      int idx = intern(cur().text, line);
      Symbol& sym = m_.symbols[static_cast<std::size_t>(idx)];
      sym.lo = lo;
      sym.up = up;
      // Integrality is sticky: a later re-declaration (e.g. `Positive
      // Variables i1;` narrowing the box of an `Integer` symbol) must not
      // clear kind back to Continuous, or the symbol silently stops being
      // integral with no diagnostic anywhere.
      if (kind != dag::VarKind::Continuous) {
        sym.kind = kind;
        sym.integral_line = line;
      }
      advance();
      if (cur().kind == Tok::Comma) { advance(); continue; }
      break;
    }
    expect(Tok::Semi, "';' after a variable declaration");
  }

  void declare_equations()
  {
    advance();
    while (true) {
      if (cur().kind != Tok::Ident) throw ParseError(cur().line, "equation name");
      declared_equations_.insert(cur().text);
      advance();
      if (cur().kind == Tok::Comma) { advance(); continue; }
      break;
    }
    expect(Tok::Semi, "';' after an equation declaration");
  }

  void equation_definition()
  {
    Equation eq;
    eq.name = cur().text;
    eq.line = cur().line;
    advance();
    expect(Tok::DotDot, "'..' after an equation name");

    eq.lhs = expression();
    if (cur().kind != Tok::Rel) {
      throw ParseError(cur().line, "expected =E=, =L=, =G= or =N=");
    }
    eq.rel = cur().rel;
    advance();
    eq.rhs = expression();
    expect(Tok::Semi, "';' after an equation definition");

    m_.equations.push_back(std::move(eq));
  }

  void attribute_assignment()
  {
    int line = cur().line;
    std::string name = cur().text;
    advance();
    expect(Tok::Dot, "'.'");
    if (cur().kind != Tok::Ident) throw ParseError(line, "attribute name");
    std::string attr = cur().text;
    advance();

    auto it = m_.symbol_index.find(name);
    if (it == m_.symbol_index.end()) {
      // Model attributes (`m.limrow`), equation marginals, solver hints: not
      // ours to interpret, and erroring here would reject valid files.
      skip_to_semi();
      return;
    }
    if (attr != "lo" && attr != "up" && attr != "fx" && attr != "l") {
      skip_to_semi();
      return;
    }

    expect(Tok::Assign, "'=' in an attribute assignment");
    int value_node = expression();
    auto value = m_.ast.fold(value_node);
    if (!value) {
      throw ParseError(line, "'" + name + "." + attr + "' must be a constant");
    }
    expect(Tok::Semi, "';' after an attribute assignment");

    Symbol& s = m_.symbols[static_cast<std::size_t>(it->second)];
    if (attr == "lo") s.lo = *value;
    else if (attr == "up") s.up = *value;
    else if (attr == "fx") { s.fixed = true; s.fx = *value; }
    else { s.has_level = true; s.level = *value; }
  }

  void solve_statement()
  {
    int line = cur().line;
    while (cur().kind != Tok::Semi && cur().kind != Tok::End) {
      if (is_word("minimizing") || is_word("minimising") || is_word("maximizing")
          || is_word("maximising"))
      {
        bool maximise = cur().text[1] == 'a';
        advance();
        if (cur().kind != Tok::Ident) {
          throw ParseError(cur().line, "objective variable name after minimizing/maximizing");
        }
        auto it = m_.symbol_index.find(cur().text);
        if (it == m_.symbol_index.end()) {
          throw ParseError(cur().line, "undeclared objective variable '" + cur().text + "'");
        }
        m_.objvar = it->second;
        m_.symbols[static_cast<std::size_t>(m_.objvar)].is_objvar = true;
        m_.maximise = maximise;
        m_.have_solve = true;
      }
      advance();
    }
    if (cur().kind == Tok::Semi) advance();
    if (!m_.have_solve) {
      // `Solve m using CNS;` -- a constrained nonlinear system. There is no
      // objective to minimise, and Problem has nowhere to put "feasibility
      // only".
      throw ParseError(line,
                       "Solve statement names no objective (a feasibility-only "
                       "model has no form in Problem)",
                       ErrorKind::Unrepresentable);
    }
  }

  // -- expressions ----------------------------------------------------------

  auto expression() -> int
  {
    int lhs = term();
    while (cur().kind == Tok::Plus || cur().kind == Tok::Minus) {
      bool plus = cur().kind == Tok::Plus;
      advance();
      int rhs = term();
      lhs = plus ? m_.ast.add(lhs, rhs) : m_.ast.sub(lhs, rhs);
    }
    return lhs;
  }

  auto term() -> int
  {
    int lhs = unary();
    while (cur().kind == Tok::Star || cur().kind == Tok::Slash) {
      bool times = cur().kind == Tok::Star;
      advance();
      int rhs = unary();
      lhs = times ? m_.ast.mul(lhs, rhs) : m_.ast.div(lhs, rhs);
    }
    return lhs;
  }

  /// Unary sign binds *looser* than `**`: `-x**2` is `-(x**2)`.
  auto unary() -> int
  {
    if (cur().kind == Tok::Minus) { advance(); return m_.ast.neg(unary()); }
    if (cur().kind == Tok::Plus) { advance(); return unary(); }
    return power();
  }

  /// `**` is right-associative and its exponent may be signed (`x**-1`).
  auto power() -> int
  {
    int base = atom();
    if (cur().kind == Tok::Pow) {
      int line = cur().line;
      advance();
      int exponent = unary();
      return m_.ast.call(Func::Power, {base, exponent}, line);
    }
    return base;
  }

  auto atom() -> int
  {
    int line = cur().line;

    if (cur().kind == Tok::Number) {
      double v = cur().value;
      advance();
      return m_.ast.num(v, line);
    }

    if (cur().kind == Tok::LParen) {
      advance();
      int inner = expression();
      expect(Tok::RParen, "')'");
      return inner;
    }

    if (cur().kind != Tok::Ident) {
      throw ParseError(line, "expected a number, identifier or '('");
    }

    std::string name = cur().text;

    // GAMS special values, legal wherever a number is.
    if (name == "inf") { advance(); return m_.ast.num(kInf, line); }
    if (name == "eps") { advance(); return m_.ast.num(0.0, line); }
    if (name == "na" || name == "undf") {
      throw ParseError(line, "the special value '" + name + "' cannot be represented",
                       ErrorKind::Unrepresentable);
    }

    if (peek().kind == Tok::LParen) {
      auto const& supported = supported_functions();
      auto it = supported.find(name);
      if (it == supported.end()) {
        if (unsupported_functions().count(name) != 0) {
          throw ParseError(line, "function '" + name + "' is not supported yet",
                           ErrorKind::UnsupportedFunction);
        }
        throw ParseError(line, "unknown function '" + name + "'",
                         ErrorKind::UnsupportedFunction);
      }
      advance();
      advance();  // '('
      std::vector<int> args;
      args.push_back(expression());
      while (cur().kind == Tok::Comma) {
        advance();
        args.push_back(expression());
      }
      expect(Tok::RParen, "')' closing a function call");
      return build_call(it->second, std::move(args), line);
    }

    auto it = m_.symbol_index.find(name);
    if (it == m_.symbol_index.end()) {
      throw ParseError(line, "undeclared identifier '" + name + "'");
    }
    advance();
    return m_.ast.var(it->second, line);
  }

  auto build_call(Func f, std::vector<int> args, int line) -> int
  {
    if (f == Func::Min || f == Func::Max) {
      if (args.empty()) throw ParseError(line, "min/max needs at least one argument");
      int acc = args[0];
      for (std::size_t i = 1; i < args.size(); ++i) {
        acc = m_.ast.call(f, {acc, args[i]}, line);
      }
      return acc;
    }
    std::size_t want = (f == Func::Power) ? 2 : 1;
    if (args.size() != want) {
      throw ParseError(line, "wrong number of arguments to this function");
    }
    return m_.ast.call(f, std::move(args), line);
  }

  // -- symbol table ---------------------------------------------------------

  auto intern(std::string const& folded, int line) -> int
  {
    auto it = m_.symbol_index.find(folded);
    if (it != m_.symbol_index.end()) return it->second;
    Symbol s;
    s.name = folded;
    s.decl_line = line;
    m_.symbols.push_back(std::move(s));
    int idx = static_cast<int>(m_.symbols.size()) - 1;
    m_.symbol_index.emplace(folded, idx);
    return idx;
  }

  std::vector<Token> tokens_;
  std::size_t pos_ = 0;
  Model& m_;
  std::vector<Diagnostic>& warnings_;
  std::unordered_set<std::string> declared_equations_;
};

// ---------------------------------------------------------------------------
// Objective-variable elimination
// ---------------------------------------------------------------------------

struct Split {
  double coeff;
  int rem;
};

/**
 * Decompose `node` into `coeff * sym + rem`, where `rem` does not mention
 * `sym`. std::nullopt means "not linear in sym" -- a detection result, not a
 * failure to work around: the caller falls back to keeping the variable.
 *
 * The `contains` guard at the top is what keeps this short: every subtree free
 * of `sym` is handled once, generically, so the switch below only ever sees
 * the cases where `sym` actually appears.
 */
auto split(Ast& ast, int root, int sym) -> std::optional<Split>
{
  // Forward sweep rather than recursion: a child's index is always smaller than
  // its parent's, so one pass in index order computes every node's
  // decomposition from its children's. Recursion here would be depth-linear in
  // the number of terms, which overflows the stack on the larger MINLPLib
  // models.
  std::vector<std::optional<Split>> table(static_cast<std::size_t>(root) + 1);

  for (int i = 0; i <= root; ++i) {
    auto const at = [&](std::size_t k) { return static_cast<std::size_t>(k); };

    if (!ast.contains(i, sym)) {
      table[at(i)] = Split{0.0, i};  // whole subtree is remainder
      continue;
    }

    // Copied, not referenced: the Ast builders below can reallocate the node
    // vector and invalidate any reference into it.
    Kind const kind = ast[i].kind;
    std::vector<int> const args = ast[i].args;
    auto child = [&](std::size_t k) { return table[at(args[k])]; };

    switch (kind) {
      case Kind::Var:
        table[at(i)] = Split{1.0, ast.num(0.0)};
        break;

      case Kind::Neg:
        if (auto a = child(0)) table[at(i)] = Split{-a->coeff, ast.neg(a->rem)};
        break;

      case Kind::Add: {
        auto a = child(0);
        auto b = child(1);
        if (a && b) table[at(i)] = Split{a->coeff + b->coeff, ast.add(a->rem, b->rem)};
        break;
      }

      case Kind::Sub: {
        auto a = child(0);
        auto b = child(1);
        if (a && b) table[at(i)] = Split{a->coeff - b->coeff, ast.sub(a->rem, b->rem)};
        break;
      }

      case Kind::Mul: {
        // Exactly one side may mention sym, and the other must be constant.
        std::size_t const other = ast.contains(args[0], sym) ? 1 : 0;
        if (ast.contains(args[other], sym)) break;  // sym * sym
        auto k = ast.fold(args[other]);
        if (!k) break;                              // sym * x1
        if (auto a = child(1 - other)) {
          table[at(i)] = Split{*k * a->coeff, ast.mul(args[other], a->rem)};
        }
        break;
      }

      case Kind::Div: {
        if (ast.contains(args[1], sym)) break;  // sym in denominator
        auto k = ast.fold(args[1]);
        if (!k || feq(*k, 0.0)) break;
        if (auto a = child(0)) {
          table[at(i)] = Split{a->coeff / *k, ast.div(a->rem, args[1])};
        }
        break;
      }

      default:
        break;  // sym under exp/log/sqr/power/... -- nonlinear, stays nullopt
    }
  }

  return table[static_cast<std::size_t>(root)];
}

/// Returns the objective's AST index, or nullopt if no equation defines the
/// objective variable linearly (the caller then takes the fallback path).
auto eliminate_objective(Model& m, std::vector<Diagnostic>& warnings)
    -> std::optional<int>
{
  std::optional<int> objective;
  int candidates = 0;

  for (auto& eq : m.equations) {
    if (eq.rel != Rel::E) continue;
    if (!m.ast.contains(eq.lhs, m.objvar) && !m.ast.contains(eq.rhs, m.objvar)) {
      continue;
    }

    auto lhs = split(m.ast, eq.lhs, m.objvar);
    auto rhs = split(m.ast, eq.rhs, m.objvar);
    if (!lhs || !rhs) continue;

    double const coeff = lhs->coeff - rhs->coeff;
    if (std::fabs(coeff) < 1e-12) continue;  // the variable cancels

    ++candidates;
    if (objective) continue;

    // objvar = (rhs_rem - lhs_rem) / coeff
    int const remainder = m.ast.sub(rhs->rem, lhs->rem);
    objective = m.ast.div(remainder, m.ast.num(coeff));
    eq.consumed = true;
  }

  if (candidates > 1) {
    warnings.push_back(
        {0, "several equations define the objective variable; used the first"});
  }
  return objective;
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

template<typename T>
class Lowerer
{
public:
  Lowerer(Model& model, ParseOptions const& options, ParsedModel<T>& out)
      : m_(model)
      , opt_(options)
      , out_(out)
      , sym_node_(model.symbols.size(), kNoNode)
      , memo_(model.ast.size(), kNoNode)
  {
  }

  void create_variables()
  {
    bool any_level = false;
    for (auto const& s : m_.symbols) any_level = any_level || s.has_level;

    for (std::size_t i = 0; i < m_.symbols.size(); ++i) {
      Symbol const& s = m_.symbols[i];
      if (s.eliminated) continue;
      // Fixed variables were rewritten to literals in the AST, so they have no
      // Var node and take no search dimension.
      if (s.fixed && opt_.fold_fixed_to_const) continue;

      double lo = s.fixed ? s.fx : s.lo;
      double up = s.fixed ? s.fx : s.up;
      bool const discrete = s.kind != dag::VarKind::Continuous;
      bool const lo_defaulted = !std::isfinite(lo);
      bool const up_defaulted = !std::isfinite(up);
      if (lo_defaulted) lo = -opt_.default_bound;
      if (up_defaulted) up = opt_.default_bound;

      // GAMS .lo/.up assignments carry arbitrary reals; Problem::validate()
      // requires Integer/Binary bounds to be integer-valued. Snap inward
      // (never outward, which would admit values a finite .up was meant to
      // exclude) before the empty-box check below, so e.g. `i.lo=0.3;
      // i.up=0.7` -- integer-empty, but not real-empty -- is caught rather
      // than silently truncated into `problem.var` and rejected deep inside
      // validate() with no line number.
      if (discrete) {
        lo = std::ceil(lo);
        up = std::floor(up);
      }

      if (lo > up) {
        throw ParseError(s.decl_line, "'" + s.name + "' has an empty box");
      }

      // An unbounded integer defaults to the same +-default_bound as a
      // continuous variable, per this project's decision to keep one default
      // rather than a separate, smaller, integer-specific one (GAMS's own
      // classic default of 100 was deliberately not adopted here). But a
      // multi-million-wide integer domain is a search-quality cliff, not a
      // cosmetic substitution -- unbounded variables are already the largest
      // silent quality issue in the corpus -- so the warning names the
      // resulting domain size explicitly, not just the bound, so a
      // gams_report run over a corpus can grep for it.
      auto const domain_suffix = [&]() -> std::string {
        if (!discrete) return "";
        return " (integer domain size " + std::to_string(up - lo + 1.0) + ")";
      };
      if (lo_defaulted) {
        warn(s.decl_line, "'" + s.name + "' has no lower bound; using "
                               + std::to_string(lo) + domain_suffix());
      }
      if (up_defaulted) {
        warn(s.decl_line, "'" + s.name + "' has no upper bound; using "
                               + std::to_string(up) + domain_suffix());
      }

      dag::VarKind kind = s.kind;
      if (kind == dag::VarKind::Binary && (lo < 0.0 || lo > 0.0 || up < 1.0 || up > 1.0)) {
        // A binary narrowed by e.g. `.up = 0` no longer has the exact [0,1]
        // box Problem::validate() requires of VarKind::Binary specifically.
        // Integer with a domain of size 1 is the same search behaviour --
        // GreedyCompositionPolicy::fill_integer handles it via
        // integer_domain_size(), and unresolved() skips a degenerate box
        // entirely -- so nothing is lost but BinaryEnumerate's fixed fan-out
        // of 2, which is meaningless for a domain that isn't size 2 anyway.
        kind = dag::VarKind::Integer;
      }

      auto expr = out_.problem.var(static_cast<T>(lo), static_cast<T>(up), kind);
      sym_node_[i] = expr.id();
      out_.var_names.push_back(s.name);
      if (any_level) {
        double start = s.has_level ? s.level : std::clamp(0.0, lo, up);
        out_.initial_point.push_back(static_cast<T>(std::clamp(start, lo, up)));
      }
    }
  }

  /// Lower the objective first: its root id is what any surviving reference to
  /// the objective variable resolves to, which is how substitution costs no
  /// extra nodes.
  void set_objective(int objective_ast)
  {
    emit_roots({objective_ast});
    std::size_t root = memo_[static_cast<std::size_t>(objective_ast)];
    if (graph().nodes[root].op == dag::Op::Const) {
      throw ParseError(0, "the objective is constant", ErrorKind::Unrepresentable);
    }
    if (m_.maximise) {
      root = graph().emit(dag::Op::Neg, {root});
      out_.sense = Sense::Maximise;
    }
    if (m_.objvar >= 0) {
      sym_node_[static_cast<std::size_t>(m_.objvar)] = root;
    }
    out_.problem.set_objective(dag::Expr<T>(&graph(), root));
    objective_root_ = root;
  }

  void add_constraints()
  {
    // Classify first, emit second: only the sides that actually reach the DAG
    // are marked live, so a constant right-hand side never becomes a node.
    struct Pending {
      std::string name;
      dag::Cmp cmp;
      int lhs;
      int rhs;       // -1 when the right-hand side folded to a constant
      T bound;
    };

    std::vector<Pending> pending;
    std::vector<int> roots;

    for (auto const& eq : m_.equations) {
      if (eq.consumed) continue;
      if (eq.rel == Rel::N) {
        warn(eq.line, "'" + eq.name + "' is a =N= equation and asserts nothing; dropped");
        continue;
      }

      // `L =G= R` is `R =L= L`.
      int lhs = eq.lhs;
      int rhs = eq.rhs;
      dag::Cmp cmp = (eq.rel == Rel::E) ? dag::Cmp::EQ : dag::Cmp::LE;
      if (eq.rel == Rel::G) std::swap(lhs, rhs);

      auto lhs_const = m_.ast.fold(lhs);
      auto rhs_const = m_.ast.fold(rhs);

      if (lhs_const && rhs_const) {
        bool holds = (cmp == dag::Cmp::EQ) ? feq(*lhs_const, *rhs_const)
                                           : (*lhs_const <= *rhs_const);
        warn(eq.line,
             "'" + eq.name + "' has no variables and is "
                 + (holds ? "trivially satisfied; dropped"
                          : "VIOLATED -- the model as written is infeasible"));
        continue;
      }

      roots.push_back(lhs);
      if (rhs_const) {
        // Keeping a constant right-hand side out of the DAG saves one node per
        // constraint, and ConstraintRef carries it directly.
        pending.push_back({eq.name, cmp, lhs, -1, static_cast<T>(*rhs_const)});
      } else {
        roots.push_back(rhs);
        pending.push_back({eq.name, cmp, lhs, rhs, 0});
      }
    }

    emit_roots(roots);

    for (auto const& p : pending) {
      std::size_t const lhs_id = memo_[static_cast<std::size_t>(p.lhs)];
      std::size_t const root =
          (p.rhs < 0) ? lhs_id
                      : emit_binary(dag::Op::Sub, lhs_id,
                                    memo_[static_cast<std::size_t>(p.rhs)]);
      out_.problem.add_constraint(dag::Expr<T>(&graph(), root), p.cmp, p.bound);
      out_.constraint_names.push_back(p.name);
    }
  }

  /// A bound stated on an eliminated objective variable is a real constraint on
  /// the objective expression and must not vanish with the variable.
  void add_objective_variable_bounds()
  {
    if (m_.objvar < 0) return;
    Symbol const& s = m_.symbols[static_cast<std::size_t>(m_.objvar)];
    if (!s.eliminated) return;

    if (std::isfinite(s.up)) {
      out_.problem.add_constraint(dag::Expr<T>(&graph(), objective_root_),
                                  dag::Cmp::LE, static_cast<T>(s.up));
      out_.constraint_names.push_back(s.name + ".up");
    }
    if (std::isfinite(s.lo)) {
      std::size_t negated = graph().emit(dag::Op::Neg, {objective_root_});
      out_.problem.add_constraint(dag::Expr<T>(&graph(), negated), dag::Cmp::LE,
                                  static_cast<T>(-s.lo));
      out_.constraint_names.push_back(s.name + ".lo");
    }
  }

private:
  static constexpr std::size_t kNoNode = static_cast<std::size_t>(-1);

  auto graph() -> dag::ExprDAG<T>& { return out_.problem.graph; }

  void warn(int line, std::string message)
  {
    out_.warnings.push_back({line, std::move(message)});
  }

  auto emit_const(double v) -> std::size_t
  {
    dag::DAGNodePayload<T> p;
    p.constant = static_cast<T>(v);
    return graph().emit(dag::Op::Const, {}, p);
  }

  auto emit_binary(dag::Op op, std::size_t a, std::size_t b) -> std::size_t
  {
    // Mirrors Expr::operator*: x * x is Sqr, which is both cheaper and tighter
    // under interval arithmetic than treating the operands as independent.
    if (op == dag::Op::Mul && a == b) return graph().emit(dag::Op::Sqr, {a});
    return graph().emit(op, {a, b});
  }

  /**
   * @brief Emit every DAG node reachable from `roots`, filling memo_.
   *
   * Two linear passes over the index range, both leaning on the Ast invariant
   * that a child's index is smaller than its parent's: mark reachability
   * backwards from the roots, then emit forwards so every operand already has
   * an id. Deliberately not a recursive post-order walk -- MINLPLib has single
   * expressions with hundreds of thousands of terms, which blow the stack.
   */
  void emit_roots(std::vector<int> const& roots)
  {
    if (roots.empty()) return;

    int top = 0;
    for (int r : roots) top = std::max(top, r);

    auto const& folds = m_.ast.folds();
    std::vector<char> live(static_cast<std::size_t>(top) + 1, 0);
    for (int r : roots) live[static_cast<std::size_t>(r)] = 1;

    for (int i = top; i >= 0; --i) {
      if (live[static_cast<std::size_t>(i)] == 0) continue;
      // A wholly constant subtree collapses to one Const, so its children are
      // not live and never get nodes of their own.
      if (folds[static_cast<std::size_t>(i)]) continue;

      Node const& node = m_.ast[i];
      // Power's second argument is only ever emitted as a DAG operand when it
      // isn't wholly constant: a constant exponent is baked into the
      // rewrite/payload at emit_call time (see Func::Power below) rather than
      // needing a node of its own.
      std::size_t const operands =
          (node.kind == Kind::Call && node.func == Func::Power
           && folds[static_cast<std::size_t>(node.args[1])])
          ? 1
          : node.args.size();
      for (std::size_t k = 0; k < operands; ++k) {
        live[static_cast<std::size_t>(node.args[k])] = 1;
      }
    }

    memo_.resize(m_.ast.size(), kNoNode);
    for (int i = 0; i <= top; ++i) {
      if (live[static_cast<std::size_t>(i)] == 0) continue;
      if (memo_[static_cast<std::size_t>(i)] != kNoNode) continue;
      memo_[static_cast<std::size_t>(i)] = emit_node(i);
    }
  }

  auto emit_node(int n) -> std::size_t
  {
    // Any wholly constant subtree becomes a single Const node. Besides being
    // smaller, this guarantees no unary Op ever lands directly on a Const,
    // which GraphBuilder rejects.
    if (auto v = m_.ast.fold(n)) return emit_const(*v);

    Node const& node = m_.ast[n];
    auto in = [&](std::size_t k) {
      return memo_[static_cast<std::size_t>(node.args[k])];
    };

    switch (node.kind) {
      case Kind::Num:
        return emit_const(node.value);

      case Kind::Var: {
        std::size_t id = sym_node_[static_cast<std::size_t>(node.sym)];
        if (id == kNoNode) {
          throw ParseError(node.line,
                           "'" + m_.symbols[static_cast<std::size_t>(node.sym)].name
                               + "' is used but has no value",
                           ErrorKind::Unrepresentable);
        }
        return id;
      }

      case Kind::Add: return emit_binary(dag::Op::Add, in(0), in(1));
      case Kind::Sub: return emit_binary(dag::Op::Sub, in(0), in(1));
      case Kind::Mul: return emit_binary(dag::Op::Mul, in(0), in(1));
      case Kind::Div: return emit_binary(dag::Op::Div, in(0), in(1));
      case Kind::Neg: return graph().emit(dag::Op::Neg, {in(0)});
      case Kind::Call: return emit_call(n);
    }
    throw ParseError(node.line, "unhandled expression kind");
  }

  auto emit_call(int n) -> std::size_t
  {
    Node const& node = m_.ast[n];
    int const line = node.line;
    Func const f = node.func;
    std::vector<int> const args = node.args;

    auto operand = [&](std::size_t k) {
      return memo_[static_cast<std::size_t>(args[k])];
    };
    auto unary = [&](dag::Op op) { return graph().emit(op, {operand(0)}); };

    switch (f) {
      case Func::Sqr:  return unary(dag::Op::Sqr);
      case Func::Sqrt: return unary(dag::Op::Sqrt);
      case Func::Exp:  return unary(dag::Op::Exp);
      case Func::Log:  return unary(dag::Op::Log);
      case Func::Abs:  return unary(dag::Op::Abs);
      case Func::Sin:  return unary(dag::Op::Sin);
      case Func::Cos:  return unary(dag::Op::Cos);
      case Func::Tanh: return unary(dag::Op::Tanh);

      case Func::Log10:
      case Func::Log2: {
        std::size_t l = unary(dag::Op::Log);
        double scale = 1.0 / ((f == Func::Log10) ? std::log(10.0) : std::log(2.0));
        return emit_binary(dag::Op::Mul, l, emit_const(scale));
      }

      case Func::Min: return emit_binary(dag::Op::Min, operand(0), operand(1));
      case Func::Max: return emit_binary(dag::Op::Max, operand(0), operand(1));

      case Func::Power: {
        auto exponent = m_.ast.fold(args[1]);
        if (!exponent) {
          // A non-constant exponent is still GAMS's own `**`: exp(y*log(x)),
          // with y now a DAG operand instead of a payload constant. Same
          // domain requirement (x > 0) as the constant case below, and the
          // same reasoning: this is what `**` literally means, not a
          // rewrite chosen for convenience. `emit_roots` above only marks
          // args[1] live (so `operand(1)` is valid here) when it isn't
          // foldable, matching this branch exactly.
          std::size_t log_base = unary(dag::Op::Log);
          std::size_t scaled = emit_binary(dag::Op::Mul, log_base, operand(1));
          return graph().emit(dag::Op::Exp, {scaled});
        }
        // `x**0.5` is sqrt exactly -- same function, same interval enclosure,
        // one op. This is a rename, not the kind of lossy rewrite refused
        // above, and MINLPLib writes Euclidean norms this way.
        if (feq(*exponent, 0.5)) return unary(dag::Op::Sqrt);

        // An exact integer exponent that fits in PowN's `int` payload gets
        // Op::PowN's tighter (repeated-squaring) interval enclosure.
        // floor(x) <= x always, so !(floor(x) < x) is "already
        // integer-valued" without an -Wfloat-equal-triggering ==.
        if (!(std::floor(*exponent) < *exponent)
            && std::fabs(*exponent) <= 1e9)
        {
          dag::DAGNodePayload<T> p;
          p.int_exp = static_cast<int>(*exponent);
          return graph().emit(dag::Op::PowN, {operand(0)}, p);
        }

        // Any other constant exponent: lower to exp(c * log(x)), exactly
        // GAMS's own definition of `**` (see the domain note above this
        // function/§4.4.1 of design/GAMS_FRONTEND.md) rather than a widening
        // -- it needs x > 0, same as a literal reading of `**` already does.
        std::size_t log_base = unary(dag::Op::Log);
        std::size_t scaled =
            emit_binary(dag::Op::Mul, log_base, emit_const(*exponent));
        return graph().emit(dag::Op::Exp, {scaled});
      }
    }
    throw ParseError(line, "unhandled function");
  }

  Model& m_;
  ParseOptions const& opt_;
  ParsedModel<T>& out_;
  std::vector<std::size_t> sym_node_;
  std::vector<std::size_t> memo_;
  std::size_t objective_root_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

template<typename T>
auto parse(std::string_view source, ParseOptions const& options) -> ParsedModel<T>
{
  ParsedModel<T> out;
  Model m;

  Parser parser(lex(source), m, out.warnings);
  parser.run();

  if (!m.have_solve) {
    throw ParseError(0, "the file has no Solve statement", ErrorKind::Unrepresentable);
  }

  if (options.reject_discrete) {
    for (auto const& s : m.symbols) {
      if (s.kind != dag::VarKind::Continuous) {
        throw ParseError(s.integral_line,
                         "'" + s.name + "' is discrete; reject_discrete is set",
                         ErrorKind::Discrete);
      }
    }
  }

  // Fixed variables become literals before anything else looks at the tree, so
  // constant folding sees through them.
  if (options.fold_fixed_to_const) {
    std::vector<std::optional<double>> literal(m.symbols.size());
    bool any = false;
    for (std::size_t i = 0; i < m.symbols.size(); ++i) {
      if (m.symbols[i].fixed && static_cast<int>(i) != m.objvar) {
        literal[i] = m.symbols[i].fx;
        any = true;
      }
    }
    if (any) m.ast.substitute_literals(literal);
  }

  auto objective = eliminate_objective(m, out.warnings);
  if (objective) {
    m.symbols[static_cast<std::size_t>(m.objvar)].eliminated = true;
  } else {
    out.warnings.push_back(
        {0, "could not solve any equation for '"
                + m.symbols[static_cast<std::size_t>(m.objvar)].name
                + "'; keeping it as a variable, which costs one search dimension"});
    objective = m.ast.var(m.objvar, 0);
  }

  Lowerer<T> lowerer(m, options, out);
  lowerer.create_variables();
  lowerer.set_objective(*objective);
  lowerer.add_constraints();
  lowerer.add_objective_variable_bounds();

  // Every structural invariant validate() checks is either guaranteed by how
  // the lowerer builds the DAG (topology, op_count, bare-Const roots) or by
  // Lowerer::create_variables (integer-valued/[0,1] boxes, non-empty boxes --
  // see the snapping there). This call is what actually proves that, rather
  // than trusting it: any throw here is a frontend bug, not a malformed input
  // that should have been caught earlier as a ParseError with a better
  // message.
  try {
    out.problem.validate();
  } catch (cuminlp::error const& e) {
    throw ParseError(0, e.what(), ErrorKind::Unrepresentable);
  }

  return out;
}

template<typename T>
auto parse_file(std::filesystem::path const& path, ParseOptions const& options)
    -> ParsedModel<T>
{
  std::ifstream in(path);
  if (!in) throw ParseError(0, "cannot open '" + path.string() + "'", ErrorKind::Io);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  std::string const text = buffer.str();
  return parse<T>(text, options);
}

template auto parse<double>(std::string_view, ParseOptions const&) -> ParsedModel<double>;
template auto parse<float>(std::string_view, ParseOptions const&) -> ParsedModel<float>;
template auto parse_file<double>(std::filesystem::path const&, ParseOptions const&)
    -> ParsedModel<double>;
template auto parse_file<float>(std::filesystem::path const&, ParseOptions const&)
    -> ParsedModel<float>;

}
