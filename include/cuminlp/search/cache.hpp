#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"
#include "cuminlp/region/fan_out.hpp"
#include "cuminlp/report/observer.hpp"

// The driver's backend roles, kept between iterations
//
// In the search layer, not the backend's: the eviction policy is a memory
// decision the *driver's* budget owns.
namespace cuminlp::search
{

/**
 * @brief Roles per Composition, built on first use and bounded by eviction.
 *
 * Lazy because the reachable Compositions can be numerous and most go unused
 * for a given problem: eagerly building every one of them could waste
 * enormous device memory on graphs nothing ever launches.
 *
 * Lazy *per role*, and not only per composition, for the same reason. A
 * Composition reached by the fathom path needs only an enumerator, and one
 * reached by the subdividing path needs only a sampler and a bounder; the
 * same Composition is commonly reached both ways on any problem with more
 * variables than slots, but never in the same iteration. Building all three
 * on first sight would charge the device for whichever set that iteration
 * did not use.
 */
template<typename T>
class BackendCache
{
public:
  BackendCache(std::shared_ptr<const backend::RegionBackendFactory<T>> factory,
               const model::Problem<T>& problem,
               region::FanOutSpec fan_out,
               std::size_t budget_bytes,
               std::size_t samples_per_region,
               report::SearchObserver& observer)
      : factory_(std::move(factory))
      , problem_(problem)
      , fan_out_(fan_out)
      , budget_ {budget_bytes, samples_per_region}
      , observer_(observer)
  {
    if (factory_ == nullptr) {
      throw cuminlp::InvalidConfiguration(
          "BackendCache requires a non-null RegionBackendFactory");
    }
  }

  /// Sampler and bounder for `composition`: what the subdividing path needs.
  backend::SubdivisionBundle<T>& subdivision(
      const region::Composition& composition)
  {
    return find_or_build(
        subdivisions_, composition, backend::RoleRequest::subdividing());
  }

  /**
   * @brief Enumerator for `composition`, or null when there is none.
   *
   * Null for a composition that is not fully enumerable and for a backend
   * whose `capabilities().exact_enumeration` is false -- in both cases the
   * driver simply does not take the fathom branch, which changes performance
   * rather than answers.
   */
  backend::RegionEnumerator<T>* enumerator(
      const region::Composition& composition)
  {
    if (!region::is_fully_enumerable(composition)
        || !factory_->capabilities().exact_enumeration)
    {
      return nullptr;
    }
    return find_or_build(
               enumerators_, composition, backend::RoleRequest::enumerating())
        .enumerator.get();
  }

  /// Compositions whose roles were given back to make room for others.
  std::size_t evictions() const { return evictions_; }

private:
  struct Entry
  {
    region::Composition composition;
    backend::SubdivisionBundle<T> bundle;
    std::size_t stamp = 0;  ///< use_clock_ at the last hit
  };

  backend::SubdivisionBundle<T>& find_or_build(
      std::vector<Entry>& entries,
      const region::Composition& composition,
      backend::RoleRequest roles)
  {
    for (Entry& e : entries) {
      if (e.composition == composition) {
        e.stamp = ++use_clock_;
        return e.bundle;
      }
    }
    // Retry, evicting until it fits. Only when this cache is sizing against
    // whatever the device has free (budget_.bytes == 0): under an explicit
    // per-build budget, freeing memory cannot change the verdict, so evicting
    // would cost the rebuilds and still rethrow.
    for (;;) {
      try {
        backend::SubdivisionBundle<T> built = factory_->build_subdivision(
            problem_, composition, fan_out_, budget_, roles);
        entries.push_back(Entry {composition, std::move(built), ++use_clock_});
        return entries.back().bundle;
      } catch (const cuminlp::ResourceExhausted&) {
        if (budget_.bytes != 0 || !evict_lru()) {
          throw;
        }
      }
    }
  }

  /// Give one cached Composition's device memory back, least recently used
  /// first, across both kinds of entry. Returns false when there is nothing
  /// left to give.
  bool evict_lru()
  {
    std::size_t oldest = std::numeric_limits<std::size_t>::max();
    std::vector<Entry>* from = nullptr;
    std::size_t idx = 0;
    auto scan = [&](std::vector<Entry>& entries)
    {
      for (std::size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].stamp < oldest) {
          oldest = entries[i].stamp;
          from = &entries;
          idx = i;
        }
      }
    };
    scan(subdivisions_);
    scan(enumerators_);
    if (from == nullptr) {
      return false;
    }
    from->erase(from->begin() + static_cast<std::ptrdiff_t>(idx));
    ++evictions_;
    // ConsoleReporter prints only the first of these; the counter above
    // still tallies every eviction (design/MODULE_REFACTOR.md §8).
    observer_.on_cache_eviction();
    return true;
  }

  std::shared_ptr<const backend::RegionBackendFactory<T>> factory_;
  const model::Problem<T>& problem_;
  region::FanOutSpec fan_out_;
  backend::BuildBudget budget_;
  report::SearchObserver& observer_;

  // Two rosters, one clock: eviction is least-recently-used across every
  // role the cache holds, not within each kind separately.
  std::vector<Entry> subdivisions_;
  std::vector<Entry> enumerators_;
  std::size_t use_clock_ = 0;
  std::size_t evictions_ = 0;
};

}  // namespace cuminlp::search
