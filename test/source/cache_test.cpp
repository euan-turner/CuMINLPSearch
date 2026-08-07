// Host-only tests for search/cache.hpp's BackendCache, in particular the
// graphs_built()/graphs_rebuilt() tally telemetry needs
// (design/TELEMETRY.md §4.4, §7). BackendCache names only backend::backend.hpp
// and region::Composition/FanOutSpec, so a fake RegionBackendFactory<double>
// exercises it with no device: each stub role holds a UnitToken that charges
// a shared UnitCounter on construction and refunds it on destruction, so a
// small capacity makes BackendCache's eviction path (and hence a rebuild)
// reachable without any real device memory.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/cache.hpp"

using cuminlp::backend::BackendCapabilities;
using cuminlp::backend::BoundResult;
using cuminlp::backend::BuildBudget;
using cuminlp::backend::CandidateResult;
using cuminlp::backend::Region;
using cuminlp::backend::RegionBackendFactory;
using cuminlp::backend::RegionBounder;
using cuminlp::backend::RegionCostModel;
using cuminlp::backend::RegionEnumerator;
using cuminlp::backend::RegionSampler;
using cuminlp::backend::RoleRequest;
using cuminlp::backend::SubdivisionBundle;
using cuminlp::model::Problem;
using cuminlp::region::Composition;
using cuminlp::region::FanOutSpec;
using cuminlp::region::SlotKind;
using cuminlp::search::BackendCache;

namespace
{

// How many fake "device units" are outstanding across every stub role built
// so far; a build that would push this past `capacity` fails, exactly as a
// real device build fails over its byte budget.
struct UnitCounter
{
  explicit UnitCounter(std::size_t cap)
      : capacity(cap)
  {
  }
  std::size_t capacity;
  std::size_t outstanding = 0;
};

// RAII charge against a UnitCounter: one per stub role, so evicting a bundle
// (which destroys its roles) refunds exactly what building it charged.
class UnitToken
{
public:
  explicit UnitToken(std::shared_ptr<UnitCounter> counter)
      : counter_(std::move(counter))
  {
    ++counter_->outstanding;
  }
  ~UnitToken() { --counter_->outstanding; }
  UnitToken(const UnitToken&) = delete;
  UnitToken& operator=(const UnitToken&) = delete;

private:
  std::shared_ptr<UnitCounter> counter_;
};

class StubSampler : public RegionSampler<double>
{
public:
  explicit StubSampler(std::shared_ptr<UnitCounter> counter)
      : token_(std::move(counter))
  {
  }
  CandidateResult<double> sample(const Region<double>&,
                                 std::uint64_t) override
  {
    return CandidateResult<double> {};
  }
  std::size_t n_samples() const override { return 1; }

private:
  UnitToken token_;
};

class StubBounder : public RegionBounder<double>
{
public:
  explicit StubBounder(std::shared_ptr<UnitCounter> counter)
      : token_(std::move(counter))
  {
  }
  BoundResult<double> bound(const Region<double>&) override
  {
    return BoundResult<double> {feasible_, obj_lb_, 1};
  }
  std::size_t n_regions() const override { return 1; }

private:
  UnitToken token_;
  std::vector<unsigned char> feasible_ {1};
  std::vector<double> obj_lb_ {0.0};
};

class StubEnumerator : public RegionEnumerator<double>
{
public:
  explicit StubEnumerator(std::shared_ptr<UnitCounter> counter)
      : token_(std::move(counter))
  {
  }
  CandidateResult<double> enumerate(const Region<double>&) override
  {
    return CandidateResult<double> {};
  }
  std::size_t n_points() const override { return 1; }

private:
  UnitToken token_;
};

// Builds whichever roles are requested, charging one unit each against
// `counter_`; throws ResourceExhausted rather than let outstanding exceed
// capacity, so BackendCache's evict-and-retry path is reachable by shrinking
// the counter's capacity instead of by starving an actual GPU.
class FakeFactory : public RegionBackendFactory<double>
{
public:
  explicit FakeFactory(std::shared_ptr<UnitCounter> counter)
      : counter_(std::move(counter))
  {
  }

  BackendCapabilities capabilities() const override
  {
    return BackendCapabilities {/*exact_enumeration=*/true,
                                /*deterministic_sampling=*/true};
  }

  RegionCostModel cost_model(const Problem<double>&) const override
  {
    return RegionCostModel {};
  }

  SubdivisionBundle<double> build_subdivision(
      const Problem<double>&,
      const Composition& composition,
      const FanOutSpec&,
      const BuildBudget&,
      RoleRequest roles) const override
  {
    std::size_t const cost = (roles.sampler ? 1u : 0u)
        + (roles.bounder ? 1u : 0u) + (roles.enumerator ? 1u : 0u);
    if (counter_->outstanding + cost > counter_->capacity) {
      throw cuminlp::ResourceExhausted(
          "FakeFactory: out of fake device units");
    }
    SubdivisionBundle<double> bundle;
    if (roles.sampler) {
      bundle.sampler = std::make_unique<StubSampler>(counter_);
    }
    if (roles.bounder) {
      bundle.bounder = std::make_unique<StubBounder>(counter_);
    }
    if (roles.enumerator && cuminlp::region::is_fully_enumerable(composition))
    {
      bundle.enumerator = std::make_unique<StubEnumerator>(counter_);
    }
    return bundle;
  }

private:
  std::shared_ptr<UnitCounter> counter_;
};

Composition continuous_comp()
{
  return Composition {.kinds = {SlotKind::Continuous}};
}

Composition binary_comp()
{
  return Composition {.kinds = {SlotKind::BinaryEnumerate}};
}

}  // namespace

TEST_CASE("subdivision() builds two graphs, one per non-null role",
          "[cache][telemetry]")
{
  auto counter = std::make_shared<UnitCounter>(1000);
  auto factory = std::make_shared<FakeFactory>(counter);
  Problem<double> problem;
  cuminlp::report::SearchObserver observer;

  BackendCache<double> cache(
      factory, problem, FanOutSpec {2}, 0, 1, false, observer);
  cache.subdivision(continuous_comp());

  CHECK(cache.graphs_built() == 2);
  CHECK(cache.graphs_rebuilt() == 0);
}

TEST_CASE("a second request for the same composition is a cache hit",
          "[cache][telemetry]")
{
  auto counter = std::make_shared<UnitCounter>(1000);
  auto factory = std::make_shared<FakeFactory>(counter);
  Problem<double> problem;
  cuminlp::report::SearchObserver observer;

  BackendCache<double> cache(
      factory, problem, FanOutSpec {2}, 0, 1, false, observer);
  Composition const comp = continuous_comp();
  cache.subdivision(comp);
  cache.subdivision(comp);

  CHECK(cache.graphs_built() == 2);  // not 4: the second call hit the cache
  CHECK(cache.evictions() == 0);
}

TEST_CASE("enumerator() tallies its own graph separately from subdivision()",
          "[cache][telemetry]")
{
  auto counter = std::make_shared<UnitCounter>(1000);
  auto factory = std::make_shared<FakeFactory>(counter);
  Problem<double> problem;
  cuminlp::report::SearchObserver observer;

  BackendCache<double> cache(
      factory, problem, FanOutSpec {2}, 0, 1, false, observer);
  cache.subdivision(continuous_comp());  // 2 graphs
  cache.enumerator(binary_comp());  // 1 more

  CHECK(cache.graphs_built() == 3);
  CHECK(cache.graphs_rebuilt() == 0);
}

TEST_CASE(
    "a composition built again after eviction counts as a rebuild, once "
    "per role",
    "[cache][telemetry]")
{
  // Capacity for exactly one subdivision bundle (sampler + bounder) at a
  // time, so a second distinct composition forces BackendCache to evict the
  // first before it can build.
  auto counter = std::make_shared<UnitCounter>(2);
  auto factory = std::make_shared<FakeFactory>(counter);
  Problem<double> problem;
  cuminlp::report::SearchObserver observer;

  // budget_bytes == 0: BackendCache only evicts when sizing against "free
  // device memory" rather than an explicit per-build budget (cache.hpp's
  // find_or_build).
  BackendCache<double> cache(
      factory, problem, FanOutSpec {2}, 0, 1, false, observer);

  Composition const comp_a {.kinds = {SlotKind::Continuous}};
  Composition const comp_b {.kinds = {SlotKind::IntegerPartition}};

  cache.subdivision(comp_a);  // 2 units used; graphs_built == 2
  REQUIRE(cache.graphs_built() == 2);
  REQUIRE(cache.evictions() == 0);

  // comp_b doesn't fit alongside comp_a under a 2-unit cap: comp_a is
  // evicted (LRU) to make room, then comp_b builds.
  cache.subdivision(comp_b);
  CHECK(cache.evictions() == 1);
  CHECK(cache.graphs_built() == 4);
  CHECK(cache.graphs_rebuilt() == 0);  // comp_b's roles are new

  // comp_a doesn't fit alongside comp_b either: comp_b is evicted, and
  // rebuilding comp_a's (composition, role) pairs -- already in ever_built_
  // from the first build -- counts as a rebuild.
  cache.subdivision(comp_a);
  CHECK(cache.evictions() == 2);
  CHECK(cache.graphs_built() == 6);
  CHECK(cache.graphs_rebuilt() == 2);
}
