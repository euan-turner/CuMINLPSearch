// Measure what subdivision buys: how much tighter the interval hull over N
// subdomains is than interval analysis on the parent box alone, and how the
// per-subregion bounds it is reduced from are distributed.
//
//   refinement_study [flags] <model.gms>
//
// design/REFINEMENT_STUDY.md. Not a solver: it runs a designed sweep over
// (parent-box width, placement, N) and writes one CSV row per cell. The
// instrument is AggregateBounderReplay at k = 1 -- the aggregate backend's own
// graph, whose single reduction over N subregions is exactly the hull under
// study -- with per-subregion bounds retained (§4, stage 1), which is a mode
// no solve ever enables.
//
// CSV rather than the PARAMS/RESULT grammar tools/minlp_status.py scrapes:
// this produces a distribution over a designed sweep, not one verdict per
// instance, and forcing it into the status-line shape would misrepresent it as
// a solver result (§5).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "cuminlp/backend/aggregate/replay.cuh"
#include "cuminlp/gams.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/bisection_budget.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/study/distribution.hpp"
#include "cuminlp/study/parent_box.hpp"
#include "cuminlp/study/refinement.hpp"
#include "cuminlp/study/uniform_partition.hpp"

namespace
{

using cuminlp::backend::aggregate::AggregateBounderReplay;
using cuminlp::model::Problem;
using cuminlp::policy::BisectionBudgetCompositionPolicy;
using cuminlp::region::SlotAssignment;
using cuminlp::study::Distribution;
using cuminlp::study::Hull;
using cuminlp::study::UniformCompositionPolicy;
using Box = std::vector<cu::interval<double>>;

/// Which partitioning rule the ladder index `budget` indexes
/// (design/REFINEMENT_STUDY.md §3.2 and §7.2).
///
/// These are not interchangeable and the choice decides what the sweep can
/// conclude. `Uniform` splits every live dimension the same `k` times, which
/// is the only shape Theorem 6.1's Definition 6.2 is stated over, but its
/// region count is exponential in the live dimension so it reaches only
/// low-dimensional instances. `Budget` is the solver's own rule and spends a
/// shared `B` on the widest live domain, so its numbers transfer to a real
/// search but its `N` is not the theorem's `N` (§7.2).
enum class Partition
{
  Uniform,
  Budget,
};

struct Options
{
  std::string model;
  Partition partition = Partition::Uniform;
  bool relative_width = false;  ///< Budget partition only: switches the
                                 ///< greedy heap to relative remaining width
  std::size_t sigma_rungs = 6;
  std::size_t reps = 8;
  std::size_t max_budget = 0;  ///< 0 => fit to the device; measured in k, the
                                ///< per-dimension bisection count
                                ///< (design/REFINEMENT_STUDY.md §7.2)
  double fixed_width = 0.0;  ///< 0 => off, use the sigma ladder (§3.1);
                              ///< > 0 => §7.3's fixed-absolute-width family
                              ///< instead, `reps` random placements
  double center_range =
      std::numeric_limits<double>::infinity();  ///< §7.3: only meaningful
                                                  ///< with fixed_width > 0;
                                                  ///< keeps placements near
                                                  ///< the origin instead of
                                                  ///< anywhere in a
                                                  ///< default-padded root
  std::size_t max_regions = 0;  ///< 0 => no cap; otherwise end the ladder at
                                 ///< the last rung with N <= this. Unlike
                                 ///< --max-budget, which means a different
                                 ///< thing under each partition (k per
                                 ///< dimension vs. B outright), this caps the
                                 ///< quantity that actually costs time, and
                                 ///< so caps it identically across instances
                                 ///< of differing live dimension.
  std::uint64_t seed = 20260812;
  std::size_t histogram_bins = 64;
  bool dump_subregions = false;
  std::string out;
};

/// §5's refusal threshold for raw dumps: 2^16 rows per launch is already a
/// large CSV, and the sweep has hundreds of launches.
constexpr std::size_t kMaxDumpRegions = 1u << 16;

auto free_device_bytes() -> std::size_t
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return 0;
  }
  return free_bytes;
}

/// The same fraction of free device memory the solvers reserve, so a ladder
/// that ends here ends where a real solve's would (§3.3).
constexpr double kBudgetFraction = 0.8;

/**
 * @brief One graph per (composition spelling, N).
 *
 * `var_ids` and `widths` are per-launch data, so every placement whose policy
 * chose the same *shape* of partition shares one built graph. The sweep is
 * ordered B-outer precisely so this is hit rather than thrashed (§5).
 */
class BounderCache
{
public:
  BounderCache(const Problem<double>& problem, std::size_t budget_bytes)
      : problem_(&problem)
      , budget_bytes_(budget_bytes)
  {
  }

  AggregateBounderReplay<double>& get(
      const cuminlp::region::Composition& composition, std::size_t n)
  {
    std::string const key = cuminlp::region::spell(composition) + "#"
        + std::to_string(n);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      auto built = std::make_unique<AggregateBounderReplay<double>>(
          AggregateBounderReplay<double>::build(*problem_,
                                                composition,
                                                n,
                                                /*branch_fan_out=*/1,
                                                budget_bytes_));
      built->retain_subregion_bounds(true);
      it = cache_.emplace(key, std::move(built)).first;
    }
    return *it->second;
  }

  void clear() { cache_.clear(); }

private:
  const Problem<double>* problem_;
  std::size_t budget_bytes_;
  std::map<std::string, std::unique_ptr<AggregateBounderReplay<double>>> cache_;
};

/// What one launch produced, copied off the bounder so the caller can reduce
/// it after the next launch has overwritten the device's buffers.
struct Launch
{
  std::vector<double> lb;
  std::vector<double> ub;
  std::vector<unsigned char> feasible;
};

Launch run_launch(AggregateBounderReplay<double>& bounder,
                  const Box& box,
                  std::span<const std::size_t> var_ids,
                  std::span<const std::size_t> widths)
{
  bounder.set_domain(box, var_ids, widths);
  bounder.launch(/*stream=*/0);

  Launch out;
  auto lb = bounder.subregion_lb();
  auto ub = bounder.subregion_ub();
  auto fe = bounder.subregion_feasible();
  out.lb.assign(lb.begin(), lb.end());
  out.ub.assign(ub.begin(), ub.end());
  out.feasible.assign(fe.begin(), fe.end());
  return out;
}

// ---- CSV helpers ----------------------------------------------------------
//
// Empty rather than nan/inf for a quantity that is undefined: §2.1 and §2.2's
// rule throughout. A NaN in a CSV averages into a summary silently; an empty
// field does not.

void put(std::ostream& os, const std::optional<double>& v)
{
  os << ',';
  if (v && std::isfinite(*v)) {
    os << *v;
  }
}

void put(std::ostream& os, double v)
{
  os << ',';
  if (std::isfinite(v)) {
    os << v;
  } else if (std::isinf(v)) {
    os << (v > 0 ? "inf" : "-inf");  // a real, interesting outcome (§2.1)
  }
}

void put_size(std::ostream& os, std::size_t v)
{
  os << ',' << v;
}

/// Column names for one distribution's summary, prefixed so `lb_` and `ub_`
/// can sit side by side in one row.
void header_distribution(std::ostream& os, const std::string& p)
{
  os << ',' << p << "count" << ',' << p << "distinct_frac" << ',' << p
     << "modal_frac" << ',' << p << "iqr" << ',' << p << "mean" << ',' << p
     << "sd" << ',' << p << "nonfinite_frac" << ',' << p << "low_score" << ','
     << p << "high_score" << ',' << p << "low_frac" << ',' << p << "high_frac";
  for (std::size_t k : cuminlp::study::kIsolationRanks) {
    os << ',' << p << "low_gap_" << k;
  }
  for (std::size_t k : cuminlp::study::kIsolationRanks) {
    os << ',' << p << "high_gap_" << k;
  }
  for (double q : cuminlp::study::kQuantileLevels) {
    os << ',' << p << "q" << (q * 100.0);
  }
}

void row_distribution(std::ostream& os, const Distribution& d)
{
  put_size(os, d.count);
  put(os, d.distinct_frac);
  put(os, d.modal_frac);
  put(os, d.iqr);
  put(os, d.mean);
  put(os, d.sd);
  put(os, d.nonfinite_frac);
  put(os, d.low_score);
  put(os, d.high_score);
  put(os, d.low_frac);
  put(os, d.high_frac);
  // Every one of these loops runs over the *declared* column count, never
  // over the vector's own size. A Distribution summarising an empty set --
  // which is what an entirely-excluded parent box produces, and those are
  // common on a heavily constrained instance -- leaves its gap and quantile
  // vectors empty, and emitting `vector.size()` fields would silently write a
  // short row that every column after it in the CSV then shifts into.
  auto pad = [&](const std::vector<std::optional<double>>& v, std::size_t want)
  {
    for (std::size_t i = 0; i < want; ++i) {
      put(os, i < v.size() ? v[i] : std::nullopt);
    }
  };
  pad(d.low_gap, cuminlp::study::kIsolationCount);
  pad(d.high_gap, cuminlp::study::kIsolationCount);
  for (std::size_t i = 0; i < cuminlp::study::kQuantileCount; ++i) {
    put(os,
        i < d.quantiles.size() ? std::optional<double>(d.quantiles[i])
                               : std::nullopt);
  }
}

void write_header(std::ostream& os)
{
  // `box_id` identifies the parent box across budgets: the refinement ladder
  // is only a ladder when compared within one box (see the sweep loop).
  // `partition` names the rule `budget` indexes: the two rules give `budget`
  // incompatible meanings (§3.2 vs §7.2), so a sweep that mixes them is only
  // readable if every row says which one it came from.
  os << "instance,partition,seed,constrained,n_vars,box_id,sigma,rep,budget"
        ",n_regions"
        ",achieved_rel_width,slots,composition,padded"
        ",base_lb,base_ub"
        ",masked_lb,masked_ub,unmasked_lb,unmasked_ub"
        ",dual_gain,primal_side_gain,width_ratio"
        ",unmasked_dual_gain,unmasked_width_ratio"
        ",excluded_frac,min_ub"
        ",conc_001,conc_005,conc_025";
  header_distribution(os, "lb_");
  header_distribution(os, "ub_");
  os << '\n';
}

void escape_into(std::ostream& os, const std::string& s)
{
  bool const needs = s.find_first_of(",\"\n") != std::string::npos;
  if (!needs) {
    os << s;
    return;
  }
  os << '"';
  for (char c : s) {
    if (c == '"') {
      os << '"';
    }
    os << c;
  }
  os << '"';
}

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  auto usage = [&](std::ostream& out)
  {
    out << "usage: " << argv[0]
        << " [--partition=uniform|budget] [--relative-width]"
           " [--sigma-rungs=S] [--reps=R] [--fixed-width=W] [--max-budget=K]"
           " [--max-regions=N] [--seed=X] [--histogram-bins=K]"
           " [--dump-subregions]"
           " [--out=FILE] [-h|--help] <model.gms>\n";
  };

  auto help = [&]
  {
    usage(std::cout);
    std::cout
        << "\n"
           "Measures what subdivision buys (design/REFINEMENT_STUDY.md):\n"
           "  Q1  how much tighter the hull over N subdomains is than "
           "interval\n"
           "      analysis on the parent box, and how that scales with N.\n"
           "  Q2  whether the per-subregion bounds are all alike or have\n"
           "      isolated outliers.\n"
           "\n"
           "Partition:\n"
           "  --partition=uniform  (default) study::UniformCompositionPolicy\n"
           "                      (§7.2): every live dimension is bisected "
           "the same\n"
           "                      k times, so N = 2^(k * n_live), matching "
           "Theorem\n"
           "                      6.1's uniform subdivision (Definition 6.2) "
           "exactly.\n"
           "                      Continuous variables only, and "
           "low-dimensional\n"
           "                      instances only -- N is exponential in the "
           "live\n"
           "                      dimension count.\n"
           "  --partition=budget  policy::BisectionBudgetCompositionPolicy "
           "(§3.2),\n"
           "                      the rule gams_solve/aggregate_solve "
           "actually use:\n"
           "                      a shared budget B spent greedily on the "
           "widest\n"
           "                      live domain, so N = 2^B regardless of "
           "dimension\n"
           "                      and any instance is reachable. Its N is NOT "
           "the\n"
           "                      theorem's N (dimensions are left "
           "untouched), so\n"
           "                      this mode measures what a solver would see, "
           "not\n"
           "                      what the theorem predicts.\n"
           "  --relative-width    --partition=budget only: compare candidates "
           "by\n"
           "                      remaining width relative to their own "
           "domain.\n"
           "\n"
           "Design:\n"
           "  --sigma-rungs=S     parent-box widths 1, 1/2, ..., 1/2^S. "
           "Default 6.\n"
           "  --reps=R            random placements per rung (or, under\n"
           "                      --fixed-width, total placements). "
           "Default 8;\n"
           "                      sigma=1 always has exactly one (the root).\n"
           "  --fixed-width=W     ignore --sigma-rungs; use `reps` random\n"
           "                      placements of a box with absolute width W "
           "per\n"
           "                      live variable instead (§7.3) -- isolates "
           "the\n"
           "                      theorem's K from the root box's own "
           "width, which\n"
           "                      a sigma-relative ladder cannot do.\n"
           "  --center-range=C    only with --fixed-width: keep placements' "
           "centres\n"
           "                      within [-C, C] of the origin, clamped to "
           "the root.\n"
           "                      Unset (default) places anywhere in the "
           "root, which\n"
           "                      for a variable with a defaulted bound "
           "(e.g. GAMS's\n"
           "                      synthetic +-1000000) almost never lands "
           "near where\n"
           "                      the model's declared bound actually was.\n"
           "  --max-budget=K      largest ladder rung. Under "
           "--partition=uniform "
           "this\n"
           "                      is the per-dimension bisection count, so N "
           "tops\n"
           "                      out at 2^(K * n_live); under "
           "--partition=budget "
           "it\n"
           "                      is B itself, so N tops out at 2^K. 0 (the "
           "default)\n"
           "                      fits N to this device.\n"
           "  --max-regions=N     end the ladder at the last rung with at "
           "most N\n"
           "                      subregions. Caps the quantity that costs "
           "the time,\n"
           "                      identically across instances of differing "
           "live\n"
           "                      dimension, which --max-budget cannot do "
           "under the\n"
           "                      uniform partition. The last rung costs 2^"
           "n_live\n"
           "                      times the one before it, so this dominates "
           "runtime;\n"
           "                      capping at 2^20 moves the fitted exponent "
           "by under\n"
           "                      0.04 on the measured corpus. 0 (the "
           "default) is\n"
           "                      no cap.\n"
           "  --seed=X            placement RNG seed. Default 20260812.\n"
           "\n"
           "Output:\n"
           "  --histogram-bins=K  per-subregion histogram bins. Default 64.\n"
           "  --dump-subregions   also write raw per-subregion bounds; "
           "refuses\n"
           "                      above N = 65536.\n"
           "  --out=FILE          CSV destination. Default stdout.\n"
           "\n"
           "An undefined quantity is written as an empty field, never NaN "
           "(§2.1).\n";
  };

  Options opt;
  std::vector<std::string> positional;

  auto number = [&](const std::string& value, std::size_t& into) -> bool
  {
    try {
      std::size_t consumed = 0;
      unsigned long long const n = std::stoull(value, &consumed);
      if (consumed != value.size()) {
        return false;
      }
      into = static_cast<std::size_t>(n);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };

  auto positive_double = [&](const std::string& value, double& into) -> bool
  {
    try {
      std::size_t consumed = 0;
      double const d = std::stod(value, &consumed);
      if (consumed != value.size() || !(d > 0.0)) {
        return false;
      }
      into = d;
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };

  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    auto value_of = [&](std::string_view flag) -> std::optional<std::string>
    {
      if (arg.rfind(flag, 0) == 0 && arg.size() > flag.size()) {
        return arg.substr(flag.size());
      }
      return std::nullopt;
    };

    if (arg == "-h" || arg == "--help") {
      help();
      return 0;
    }
    if (arg == "--dump-subregions") {
      opt.dump_subregions = true;
      continue;
    }
    if (arg == "--relative-width") {
      opt.relative_width = true;
      continue;
    }
    if (auto v = value_of("--partition=")) {
      if (*v == "uniform") {
        opt.partition = Partition::Uniform;
      } else if (*v == "budget") {
        opt.partition = Partition::Budget;
      } else {
        std::cerr << "bad --partition: " << *v
                  << " (expected 'uniform' or 'budget')\n";
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--sigma-rungs=")) {
      if (!number(*v, opt.sigma_rungs)) {
        std::cerr << "bad --sigma-rungs: " << *v << '\n';
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--reps=")) {
      if (!number(*v, opt.reps) || opt.reps == 0) {
        std::cerr << "bad --reps: " << *v << '\n';
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--fixed-width=")) {
      if (!positive_double(*v, opt.fixed_width)) {
        std::cerr << "bad --fixed-width: " << *v << '\n';
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--center-range=")) {
      if (!positive_double(*v, opt.center_range)) {
        std::cerr << "bad --center-range: " << *v << '\n';
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--max-budget=")) {
      if (!number(*v, opt.max_budget)) {
        std::cerr << "bad --max-budget: " << *v << '\n';
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--max-regions=")) {
      if (!number(*v, opt.max_regions) || opt.max_regions == 0) {
        std::cerr << "bad --max-regions: " << *v << '\n';
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--histogram-bins=")) {
      if (!number(*v, opt.histogram_bins)) {
        std::cerr << "bad --histogram-bins: " << *v << '\n';
        return 2;
      }
      continue;
    }
    if (auto v = value_of("--seed=")) {
      std::size_t s = 0;
      if (!number(*v, s)) {
        std::cerr << "bad --seed: " << *v << '\n';
        return 2;
      }
      opt.seed = static_cast<std::uint64_t>(s);
      continue;
    }
    if (auto v = value_of("--out=")) {
      opt.out = *v;
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      std::cerr << "unknown flag: " << arg << '\n';
      usage(std::cerr);
      return 2;
    }
    positional.push_back(arg);
  }

  if (positional.size() != 1) {
    usage(std::cerr);
    return 2;
  }
  opt.model = positional[0];

  // Rejected rather than ignored: the uniform partition has no per-candidate
  // heap for a width comparison to apply to, so accepting the flag here would
  // silently produce rows indistinguishable from a plain uniform sweep while
  // the caller believed otherwise.
  if (opt.relative_width && opt.partition != Partition::Budget) {
    std::cerr << "--relative-width applies only to --partition=budget\n";
    return 2;
  }

  try {
    auto parsed = cuminlp::gams::parse_file<double>(opt.model);
    const Problem<double>& problem = parsed.problem;
    Box const root = problem.box_bounds;
    auto const kinds = problem.var_kinds;
    bool const constrained = !problem.constraints.empty();

    std::cerr << opt.model << ": " << parsed.var_names.size() << " variables, "
              << problem.constraints.size() << " constraints, "
              << problem.graph.nodes.size() << " DAG nodes\n";
    for (auto const& w : parsed.warnings) {
      std::cerr << "  warning (line " << w.line << "): " << w.message << '\n';
    }

    std::size_t const budget_bytes = static_cast<std::size_t>(
        static_cast<double>(free_device_bytes()) * kBudgetFraction);

    std::ofstream file;
    if (!opt.out.empty()) {
      file.open(opt.out);
      if (!file) {
        std::cerr << "cannot open --out=" << opt.out << '\n';
        return 2;
      }
    }
    std::ostream& os = opt.out.empty() ? std::cout : file;
    os.precision(17);
    write_header(os);

    // ---- the baseline: N = 1, no slots, so the one region is the box -----
    // Built once and re-set_domain'd per parent box, like every other graph
    // in the sweep.
    cuminlp::region::Composition const no_slots {};
    auto baseline_bounder = AggregateBounderReplay<double>::build(
        problem, no_slots, /*n_regions=*/1, /*branch_fan_out=*/1, budget_bytes);
    baseline_bounder.retain_subregion_bounds(true);

    BounderCache cache(problem, budget_bytes);

    // §3.3: the ladder ends where the device does. Discovered by building,
    // not predicted, so an OverBudgetError truncates the sweep cleanly rather
    // than aborting it.
    std::size_t const max_budget =
        (opt.max_budget == 0) ? 62 : opt.max_budget;

    // The parent-box family is drawn ONCE, before the budget loop, and every
    // budget is measured on the *same* boxes.
    //
    // This is load-bearing, not a tidiness point. Q1 asks how the hull
    // tightens as N grows for a given parent box; if each budget drew its own
    // boxes, `rho(N)` would be confounded with placement and the ladder would
    // not be a refinement ladder at all. An earlier version of this file
    // reseeded per budget and produced a curve that was not monotone in N --
    // which is exactly how the §3.2 invariant is supposed to catch this.
    //
    // It also makes rows joinable across budgets on `box_id`, which is what
    // any per-box analysis of the ladder needs.
    struct ParentBox
    {
      double sigma;
      std::size_t rep;
      std::size_t id;
      Box box;
    };

    std::vector<ParentBox> boxes;
    {
      std::mt19937_64 rng(opt.seed);
      std::size_t id = 0;
      if (opt.fixed_width > 0.0) {
        // §7.3: `reps` random placements of one fixed absolute width, no
        // ladder. `sigma` carries the requested width here (not a fraction)
        // -- there is only one rung, so the CSV column is constant per row,
        // same as it would be for a one-rung sigma ladder.
        for (std::size_t rep = 0; rep < opt.reps; ++rep) {
          boxes.push_back(
              ParentBox {opt.fixed_width,
                         rep,
                         id++,
                         cuminlp::study::draw_fixed_width_box<double>(
                             root, kinds, opt.fixed_width, rng,
                             opt.center_range)});
        }
      } else {
        for (double sigma : cuminlp::study::sigma_ladder(opt.sigma_rungs)) {
          std::size_t const reps =
              cuminlp::study::placements_for(sigma, opt.reps);
          for (std::size_t rep = 0; rep < reps; ++rep) {
            boxes.push_back(
                ParentBox {sigma,
                           rep,
                           id++,
                           cuminlp::study::draw_parent_box<double>(
                               root, kinds, sigma, rng)});
          }
        }
      }
    }
    std::cerr << boxes.size() << " parent boxes, budgets 1.."
              << (opt.max_budget == 0 ? std::string("device-limited")
                                      : std::to_string(opt.max_budget))
              << '\n';

    // k-outer so the graph cache is hit rather than thrashed, and so a sweep
    // truncated at large k still holds complete data for every smaller one
    // (§5).
    for (std::size_t budget = 1; budget <= max_budget; ++budget) {
      // Held through the base-class handle so the sweep body below is written
      // once: the two rules differ only in what `budget` means to them, and
      // every row records which one produced it via the `partition` column.
      std::unique_ptr<cuminlp::policy::CompositionPolicy<double>> policy;
      if (opt.partition == Partition::Uniform) {
        policy = std::make_unique<UniformCompositionPolicy<double>>(budget);
      } else {
        policy = std::make_unique<BisectionBudgetCompositionPolicy<double>>(
            budget, cuminlp::config::SearchCalibration {}, opt.relative_width);
      }

      bool budget_ok = true;

      {
        for (const ParentBox& pb : boxes) {
          if (!budget_ok) {
            break;
          }
          double const sigma = pb.sigma;
          std::size_t const rep = pb.rep;
          const Box& box = pb.box;

          SlotAssignment a;
          std::size_t n = 0;
          Launch refined;
          try {
            // policy.choose() itself can throw ResourceExhausted here: N =
            // 2^(k * n_live) overflows a 64-bit region count well before any
            // device does (§7.2), so the same catch that ends the ladder on
            // an over-budget device launch also ends it on an over-budget
            // region count.
            a = policy->choose(box, kinds);
            n = cuminlp::region::composition_fan_out(a.widths);
            // Ends the ladder rather than skipping this box, so every box
            // still has the same set of rungs and the per-box fits stay
            // joinable -- the same discipline the device-exhaustion path
            // below follows.
            if (opt.max_regions != 0 && n > opt.max_regions) {
              std::cerr << "budget " << budget << " needs N = " << n
                        << " > --max-regions=" << opt.max_regions
                        << "; stopping the ladder\n";
              budget_ok = false;
              break;
            }
            auto& bounder = cache.get(a.composition, n);
            refined = run_launch(bounder, box, a.var_ids, a.widths);
          } catch (const cuminlp::ResourceExhausted&) {
            // The ladder ends here. Everything already written stays valid;
            // nothing larger is attempted.
            std::cerr << "budget " << budget
                      << " exceeds the device or region-count budget; "
                         "stopping the ladder\n";
            budget_ok = false;
            break;
          }

          Launch const base = run_launch(baseline_bounder, box, {}, {});
          Hull<double> const baseline =
              cuminlp::study::unmasked_hull<double>(base.lb, base.ub);
          Hull<double> const masked = cuminlp::study::masked_hull<double>(
              refined.lb, refined.ub, refined.feasible);
          Hull<double> const unmasked =
              cuminlp::study::unmasked_hull<double>(refined.lb, refined.ub);

          auto const g = cuminlp::study::gains(baseline, masked);
          auto const gu = cuminlp::study::gains(baseline, unmasked);

          // §2.4: reported, never combined with anything, and paired with the
          // `constrained` column that says whether it is sound.
          auto const min_ub = cuminlp::study::min_upper_bound<double>(
              refined.ub, refined.feasible);

          Distribution const lb_dist = cuminlp::study::summarise<double>(
              refined.lb, refined.feasible, opt.histogram_bins);
          Distribution const ub_dist = cuminlp::study::summarise<double>(
              refined.ub, refined.feasible, opt.histogram_bins);

          bool const padded = !a.domain_sizes.empty()
              && a.domain_sizes != a.widths;

          escape_into(os, opt.model);
          os << ','
             << (opt.partition == Partition::Uniform
                     ? "uniform"
                     : (opt.relative_width ? "budget-relative" : "budget"))
             << ',' << opt.seed << ',' << (constrained ? 1 : 0) << ','
             << root.size() << ',' << pb.id << ',' << sigma << ',' << rep
             << ',' << budget << ',' << n << ','
             << cuminlp::study::achieved_relative_width<double>(box, root)
             << ',' << a.size() << ',';
          escape_into(os, cuminlp::region::spell(a.composition));
          os << ',' << (padded ? 1 : 0);

          put(os, baseline.lb);
          put(os, baseline.ub);
          if (masked.any) {
            put(os, masked.lb);
            put(os, masked.ub);
          } else {
            os << ",,";
          }
          put(os, unmasked.lb);
          put(os, unmasked.ub);
          put(os, g.dual_gain);
          put(os, g.primal_side_gain);
          put(os, g.width_ratio);
          put(os, gu.dual_gain);
          put(os, gu.width_ratio);
          put(os, cuminlp::study::excluded_fraction(refined.feasible));
          put(os, min_ub ? std::optional<double>(*min_ub) : std::nullopt);

          for (double eps : {0.01, 0.05, 0.25}) {
            put(os,
                masked.any ? cuminlp::study::concentration<double>(
                    refined.lb, refined.feasible, masked.lb, masked.ub, eps)
                           : 0.0);
          }

          row_distribution(os, lb_dist);
          row_distribution(os, ub_dist);
          os << '\n';

          if (opt.dump_subregions) {
            if (n > kMaxDumpRegions) {
              std::cerr << "--dump-subregions refused at N = " << n
                        << " (limit " << kMaxDumpRegions << ")\n";
            } else {
              std::ofstream dump(opt.out.empty()
                                     ? std::string("subregions")
                                     : opt.out + ".subregions");
              dump.precision(17);
              dump << "budget,sigma,rep,region,lb,ub,feasible\n";
              for (std::size_t r = 0; r < n; ++r) {
                dump << budget << ',' << sigma << ',' << rep << ',' << r << ','
                     << refined.lb[r] << ',' << refined.ub[r] << ','
                     << static_cast<int>(refined.feasible[r]) << '\n';
              }
            }
          }
        }
        if (!budget_ok) {
          break;
        }
      }

      if (!budget_ok) {
        break;
      }
      os.flush();
      // Graphs for a budget that will not recur are device memory the next,
      // larger budget needs.
      cache.clear();
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
