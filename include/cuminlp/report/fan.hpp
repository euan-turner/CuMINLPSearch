#pragma once

#include <memory>
#include <vector>

#include "cuminlp/report/observer.hpp"

// How two reporters coexist behind SearchDriver's one observer slot
// (design/TELEMETRY.md §3.3).
namespace cuminlp::report
{

/// A SearchObserver that forwards every event to a list of observers, in
/// registration order. Mechanical: no logic of its own, just the fan-out
/// that lets `--verbose` register a TelemetryReporter alongside today's
/// ConsoleReporter at the one place SearchDriver already accepts an
/// observer, so nothing else in `search` grows a second sink parameter.
class ObserverFan : public SearchObserver
{
public:
  void add(std::shared_ptr<SearchObserver> observer)
  {
    observers_.push_back(std::move(observer));
  }

  void on_start(std::size_t host_budget,
                bool host_budget_requested,
                bool compacting) override
  {
    for (auto& o : observers_) {
      o->on_start(host_budget, host_budget_requested, compacting);
    }
  }

  void on_dequeue(std::uint32_t iter, double lb) override
  {
    for (auto& o : observers_) {
      o->on_dequeue(iter, lb);
    }
  }

  void on_incumbent(double gub) override
  {
    for (auto& o : observers_) {
      o->on_incumbent(gub);
    }
  }

  void on_iteration(const IterationEvent& event) override
  {
    for (auto& o : observers_) {
      o->on_iteration(event);
    }
  }

  void on_fathom(std::size_t n_points, double gub) override
  {
    for (auto& o : observers_) {
      o->on_fathom(n_points, gub);
    }
  }

  void on_cache_eviction() override
  {
    for (auto& o : observers_) {
      o->on_cache_eviction();
    }
  }

  void on_compaction(const CompactionEvent& event) override
  {
    for (auto& o : observers_) {
      o->on_compaction(event);
    }
  }

  void on_budget_stop(const BudgetStopEvent& event) override
  {
    for (auto& o : observers_) {
      o->on_budget_stop(event);
    }
  }

  void on_backend_error(const backend::OverBudget& over) override
  {
    for (auto& o : observers_) {
      o->on_backend_error(over);
    }
  }

  void on_mid_search_error(std::string_view message) override
  {
    for (auto& o : observers_) {
      o->on_mid_search_error(message);
    }
  }

  void on_telemetry(const RunTelemetry& telemetry) override
  {
    for (auto& o : observers_) {
      o->on_telemetry(telemetry);
    }
  }

  void on_finish(const FinalReport& report) override
  {
    for (auto& o : observers_) {
      o->on_finish(report);
    }
  }

private:
  std::vector<std::shared_ptr<SearchObserver>> observers_;
};

}  // namespace cuminlp::report
