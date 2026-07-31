// Coverage report for the GAMS frontend.
//
// Runs the parser over a directory of .gms files and tallies what happened, so
// that "which operator should I add next" is a measurement rather than a guess.
//
//   gams_report [--reject-discrete] [--list] [--per-instance] <directory-or-file>...
//
// Parsed instances are counted, and their variable/constraint/node counts
// totalled; rejections are grouped by ErrorKind, with a histogram of the
// specific reasons. --list additionally names the files behind each histogram
// row, since a bare count is not something you can open and read.
//
// --per-instance instead writes one TSV row per file and nothing else. The
// aggregate above answers "what should I implement next"; the per-instance
// form answers "can this specific instance be attempted", which is what
// MINLP_STATUS.md tracks. It is a separate mode rather than an addition
// because its consumer is a script, and a summary printed around the rows
// would only have to be skipped back off.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "cuminlp/gams.hpp"

namespace gams = cuminlp::gams;

namespace
{

auto kind_name(gams::ErrorKind kind) -> char const*
{
  switch (kind) {
    case gams::ErrorKind::Syntax: return "syntax";
    case gams::ErrorKind::NonScalar: return "not-scalar-format";
    case gams::ErrorKind::Discrete: return "discrete-variables";
    case gams::ErrorKind::UnsupportedFunction: return "unsupported-function";
    case gams::ErrorKind::Unrepresentable: return "unrepresentable";
    case gams::ErrorKind::Io: return "io";
  }
  return "?";
}

/// Drop the leading "line N: " and any instance-specific identifier so that
/// messages group into a useful histogram.
auto reason(std::string message) -> std::string
{
  auto colon = message.find(": ");
  if (colon != std::string::npos) message = message.substr(colon + 2);
  auto quote = message.find('\'');
  if (quote != std::string::npos) {
    auto close = message.find('\'', quote + 1);
    if (close != std::string::npos && message.rfind("function", 0) != 0) {
      message = message.substr(0, quote) + message.substr(close + 1);
    }
  }
  return message;
}

/// Prints up to `limit` paths, one per line, then how many were elided. The
/// cap matters: "the objective is constant" covers two dozen instances and the
/// point of --list is to give you one to open, not to restate the histogram.
void list_files(std::vector<std::filesystem::path> const& files, std::size_t limit)
{
  for (std::size_t i = 0; i < files.size() && i < limit; ++i) {
    std::printf("           %s\n", files[i].c_str());
  }
  if (files.size() > limit) {
    std::printf("           ... and %zu more\n", files.size() - limit);
  }
}

struct Totals {
  int parsed = 0;
  int fallback = 0;              // objective variable could not be eliminated
  int defaulted = 0;             // at least one variable got the default box
  int defaulted_integer = 0;     // ...and at least one of those is Integer/Binary
  int defaulted_continuous_only = 0;  // ...and none of those are Integer/Binary
  int rejected = 0;
  std::size_t vars = 0;
  std::size_t constraints = 0;
  std::size_t nodes = 0;
};

}  // namespace

auto main(int argc, char** argv) -> int
{
  // --reject-discrete reproduces the pre-integrality baseline (every
  // Binary/Integer variable rejected outright), so a corpus run can be
  // A/B'd against the post-integrality default and the coverage delta
  // attributed to integrality specifically, not to anything else this
  // change touched.
  gams::ParseOptions options;
  bool list = false;
  bool per_instance = false;
  std::vector<std::filesystem::path> files;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    if (arg == "--reject-discrete") {
      options.reject_discrete = true;
      continue;
    }
    if (arg == "--list") {
      list = true;
      continue;
    }
    if (arg == "--per-instance") {
      per_instance = true;
      continue;
    }
    std::filesystem::path root(arg);
    if (std::filesystem::is_directory(root)) {
      for (auto const& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.path().extension() == ".gms") files.push_back(entry.path());
      }
    } else {
      files.push_back(root);
    }
  }
  std::sort(files.begin(), files.end());

  if (files.empty()) {
    std::fprintf(
        stderr,
        "usage: %s [--reject-discrete] [--list] [--per-instance] <directory-or-file>...\n",
        argv[0]);
    return 2;
  }

  // Named here rather than left implicit in the row order: the consumer is a
  // script, and a header it can key off survives a column being added.
  if (per_instance) {
    std::printf(
        "file\tstatus\tsense\tkind\treason\tvars\tconstraints\tnodes\tflags\n");
  }

  Totals totals;
  // Files, not counts: a histogram row you cannot trace back to an instance
  // tells you a reason is common but gives you nothing to open. The counts are
  // just .size(), so nothing is lost by storing the paths.
  std::map<std::string, std::vector<std::filesystem::path>> by_kind;
  std::map<std::string, std::vector<std::filesystem::path>> by_reason;
  std::vector<std::filesystem::path> defaulted_integer_files;

  for (auto const& file : files) {
    try {
      auto parsed = gams::parse_file<double>(file, options);
      ++totals.parsed;
      totals.vars += parsed.problem.box_bounds.size();
      totals.constraints += parsed.problem.constraints.size();
      totals.nodes += parsed.problem.graph.nodes.size();

      bool fallback = false;
      bool defaulted = false;
      bool defaulted_integer = false;
      for (auto const& w : parsed.warnings) {
        fallback = fallback || w.message.find("keeping it as a variable") != std::string::npos;
        if (w.message.find("bound; using") != std::string::npos) {
          defaulted = true;
          // create_variables() (gams/frontend.cpp) names the domain size
          // explicitly in an Integer/Binary variable's default-bound warning
          // specifically so this is greppable -- an unbounded integer is a
          // materially worse quality issue than an unbounded continuous one.
          if (w.message.find("integer domain size") != std::string::npos) {
            defaulted_integer = true;
          }
        }
      }
      totals.fallback += fallback ? 1 : 0;
      totals.defaulted += defaulted ? 1 : 0;
      totals.defaulted_integer += defaulted_integer ? 1 : 0;
      totals.defaulted_continuous_only += (defaulted && !defaulted_integer) ? 1 : 0;
      if (defaulted_integer) defaulted_integer_files.push_back(file);

      if (per_instance) {
        // Quality caveats, not failures: each of these parses and solves, so
        // it appears in no rejection table, but each is a reason to distrust
        // a bound obtained from it. The tracker needs them next to the bound.
        std::string flags;
        auto flag = [&flags](char const* name) {
          if (!flags.empty()) flags += ',';
          flags += name;
        };
        if (fallback) flag("objvar-kept");
        if (defaulted_integer) flag("default-bound-integer");
        else if (defaulted) flag("default-bound");
        // The sense is what makes "the best bound so far" an ordering rather
        // than just a number: for a maximisation a larger incumbent is the
        // better one. It is emitted per instance because the frontend has
        // already negated a Maximise objective, so nothing downstream of the
        // Problem can recover it.
        std::printf("%s\tparsed\t%s\t\t\t%zu\t%zu\t%zu\t%s\n",
                    file.filename().c_str(),
                    parsed.sense == gams::Sense::Maximise ? "max" : "min",
                    parsed.problem.box_bounds.size(),
                    parsed.problem.constraints.size(),
                    parsed.problem.graph.nodes.size(), flags.c_str());
      }
    } catch (gams::ParseError const& e) {
      ++totals.rejected;
      by_kind[kind_name(e.kind)].push_back(file);
      by_reason[reason(e.what())].push_back(file);
      if (per_instance) {
        std::printf("%s\trejected\t\t%s\t%s\t\t\t\t\n", file.filename().c_str(),
                    kind_name(e.kind), reason(e.what()).c_str());
      }
    } catch (std::exception const& e) {
      ++totals.rejected;
      by_kind["exception"].push_back(file);
      by_reason[e.what()].push_back(file);
      if (per_instance) {
        std::printf("%s\trejected\t\texception\t%s\t\t\t\t\n",
                    file.filename().c_str(), e.what());
      }
    }
  }

  if (per_instance) return 0;

  int const total = totals.parsed + totals.rejected;
  std::printf("GAMS frontend coverage over %d instances\n\n", total);
  std::printf("  parsed              %5d  (%.1f%%)\n", totals.parsed,
              100.0 * totals.parsed / total);
  std::printf("    of which:\n");
  std::printf("      objective variable eliminated  %5d\n",
              totals.parsed - totals.fallback);
  std::printf("      fell back to keeping objvar    %5d\n", totals.fallback);
  std::printf("      needed a default bound         %5d\n", totals.defaulted);
  std::printf("        of which integer/binary      %5d\n", totals.defaulted_integer);
  std::printf("        of which continuous only      %5d\n", totals.defaulted_continuous_only);
  std::printf("  rejected            %5d  (%.1f%%)\n\n", totals.rejected,
              100.0 * totals.rejected / total);

  // Sorting by descending count needs a stable tiebreak, or two reasons with
  // equal counts can swap between runs and make two reports gratuitously
  // diff-noisy. The map already orders keys, so falling back to the key gives
  // that for free.
  auto by_count_then_name = [](auto const& a, auto const& b) {
    if (a.second.size() != b.second.size()) return a.second.size() > b.second.size();
    return a.first < b.first;
  };

  std::printf("rejections by kind:\n");
  std::vector<std::pair<std::string, std::vector<std::filesystem::path>>> kinds(
      by_kind.begin(), by_kind.end());
  std::sort(kinds.begin(), kinds.end(), by_count_then_name);
  for (auto const& [name, group] : kinds) {
    std::printf("  %-22s %5zu  (%.1f%% of all)\n", name.c_str(), group.size(),
                100.0 * static_cast<double>(group.size()) / total);
  }

  // Without --list the tail is noise, so it stays capped at 25 rows. With
  // --list you are triaging, and the rare reasons are exactly the ones worth
  // seeing -- a reason with one instance is one file away from being fixed.
  std::printf("\n%s:\n", list ? "reasons" : "most common reasons");
  std::vector<std::pair<std::string, std::vector<std::filesystem::path>>> reasons(
      by_reason.begin(), by_reason.end());
  std::sort(reasons.begin(), reasons.end(), by_count_then_name);
  std::size_t const shown = list ? reasons.size() : std::min<std::size_t>(reasons.size(), 25);
  for (std::size_t i = 0; i < shown; ++i) {
    std::printf("  %5zu  %s\n", reasons[i].second.size(), reasons[i].first.c_str());
    if (list) list_files(reasons[i].second, 10);
  }

  if (list && !defaulted_integer_files.empty()) {
    // Called out separately because these instances parse and solve, so they
    // appear nowhere in the rejection tables above -- but an integer variable
    // on a 2e6-wide default box is a search-quality cliff, and any benchmark
    // built on this corpus needs to know which instances it is standing on.
    std::printf("\nparsed, but an integer/binary variable got the default bound:\n");
    list_files(defaulted_integer_files, 50);
  }

  if (totals.parsed > 0) {
    std::printf("\nparsed instances: %zu variables, %zu constraints, %zu DAG nodes in total\n",
                totals.vars, totals.constraints, totals.nodes);
  }
  return 0;
}
