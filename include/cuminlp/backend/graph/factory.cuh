#pragma once

#include <memory>

#include "cuminlp/backend/backend.hpp"
#include "cuminlp/backend/graph/cost.hpp"
#include "cuminlp/backend/graph/replay.cuh"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/region/composition.hpp"

// GraphBackendFactory: the CUDA-graph backend, as a
// backend::RegionBackendFactory. Moved out of graph_replay.cuh
// (design/MODULE_REFACTOR.md §11).
namespace cuminlp::backend::graph
{

// Unqualified names below (SlotKind, Composition, VarKind, Op::..., etc.)
// are the same ones graph_replay.cuh used unqualified before this file
// existed, when they lived in flat cuminlp:: (composition_policy.hpp,
// dag.hpp) and this content sat in namespace cuminlp::dag -- enclosing-scope
// lookup found them without qualification. Now that they live in sibling
// namespaces cuminlp::region / cuminlp::model, that lookup no longer reaches
// them, so these using-declarations restore it explicitly rather than
// re-qualifying every use-site (design/MODULE_REFACTOR.md §11 is a pure
// rename; this keeps the diff a namespace/file move, not a rewrite).
using cuminlp::region::can_fathom_without_children;
using cuminlp::region::Composition;
using cuminlp::region::composition_fan_out;
using cuminlp::region::FanOutSpec;
using cuminlp::region::is_fully_enumerable;
using cuminlp::region::slot_fan_out;
using cuminlp::region::slot_prefixes;
using cuminlp::region::SlotAssignment;
using cuminlp::region::SlotKind;
namespace decode = cuminlp::region::decode;
using cuminlp::model::Cmp;
using cuminlp::model::ConstraintRef;
using cuminlp::model::DAGNode;
using cuminlp::model::ExprDAG;
using cuminlp::model::Op;
using cuminlp::model::Problem;
using cuminlp::model::VarKind;

/**
 * @brief The CUDA-graph backend, as a backend::RegionBackendFactory.
 *
 * The one place the three replay instantiations are named together, so that
 * everything above it can ask for roles instead of for graph kinds
 * (design/MODULE_REFACTOR.md §5.3).
 *
 * One build entry point rather than §5.3's two, because the sampler here is
 * still composition-coupled: today's point graph subdivides exactly as the
 * bounder does and then draws `samples_per_region` values inside each
 * subregion. §5.2 makes the sampler a function of the problem alone, built
 * once per solve; at that point it leaves the bundle for a `build_sampler`
 * of its own and stops being charged per composition.
 */
template<typename T>
class GraphBackendFactory : public backend::RegionBackendFactory<T>
{
public:
  backend::BackendCapabilities capabilities() const override
  {
    // The exact graph is always available, and every sampler draw is a pure
    // function of (box, var_ids, salt) -- see sample_hash, which holds no
    // device-side RNG state between launches.
    return backend::BackendCapabilities {/*exact_enumeration=*/true,
                                         /*deterministic_sampling=*/true};
  }

  backend::RegionCostModel cost_model(const Problem<T>& problem) const override
  {
    return backend::graph::cost_model_for<T>(problem);
  }

  backend::SubdivisionBundle<T> build_subdivision(
      const Problem<T>& problem,
      const Composition& composition,
      const FanOutSpec& fan_out,
      const backend::BuildBudget& budget,
      backend::RoleRequest roles) const override
  {
    backend::SubdivisionBundle<T> bundle;
    if (roles.sampler) {
      bundle.sampler = std::make_unique<PointGraphReplay<T>>(
          PointGraphReplay<T>::build(problem,
                                     composition,
                                     fan_out,
                                     budget.bytes,
                                     budget.samples_per_region,
                                     budget.report_build));
    }
    // samples_per_region reaches the bounder and enumerator only so their
    // over-budget facts can cost a suggested cap against the sampler too;
    // build() clamps their own allocation to one element per region
    // regardless (GraphReplay::build, `is_point && !Exact`).
    if (roles.bounder) {
      bundle.bounder = std::make_unique<IntervalGraphReplay<T>>(
          IntervalGraphReplay<T>::build(problem,
                                        composition,
                                        fan_out,
                                        budget.bytes,
                                        budget.samples_per_region,
                                        budget.report_build));
    }
    if (roles.enumerator && is_fully_enumerable(composition)) {
      bundle.enumerator = std::make_unique<ExactGraphReplay<T>>(
          ExactGraphReplay<T>::build(problem,
                                     composition,
                                     fan_out,
                                     budget.bytes,
                                     budget.samples_per_region,
                                     budget.report_build));
    }
    return bundle;
  }
};

}  // namespace cuminlp::backend::graph
