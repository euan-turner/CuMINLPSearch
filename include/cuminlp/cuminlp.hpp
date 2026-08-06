#pragma once

// Umbrella include: the search layer and everything below it, for callers
// that want the whole solver rather than one module.
//
// This header used to define a `driver` base class holding the global bounds,
// the tolerance and the iteration budget for every concrete driver. With the
// legacy fixed-Rosenbrock driver retired there is exactly one derived class,
// so that state moved into SearchDriver itself and its accessors became
// SolveOutcome's fields (design/MODULE_REFACTOR.md §6.1). No backend is
// included here: the application chooses one (e.g. backend/graph/factory.cuh's
// GraphBackendFactory) and hands it to the driver.

#include "cuminlp/config/run_spec.hpp"
#include "cuminlp/errors.hpp"
#include "cuminlp/model/problem.hpp"
#include "cuminlp/policy/greedy.hpp"
#include "cuminlp/report/observer.hpp"
#include "cuminlp/search/cache.hpp"
#include "cuminlp/search/driver.hpp"
