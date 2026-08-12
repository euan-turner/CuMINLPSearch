# ---- Dependencies ----

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

# ---- Tests ----

# tools/minlp_status.py's log scraper: what stands between a driver summary
# line and a permanent row in MINLP_STATUS.md. Its own language, so its own
# runner -- registered with CTest so it runs with everything else rather than
# only when someone remembers it. Skipped, not failed, where there is no
# interpreter: nothing else here needs one.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
      NAME minlp_status_test
      COMMAND Python3::Interpreter
              "${CMAKE_CURRENT_SOURCE_DIR}/tools/minlp_status_test.py"
  )
else()
  message(STATUS "No Python 3 interpreter: skipping minlp_status_test")
endif()

# GAMS frontend tests: expression parsing, objective-variable elimination, and
# agreement with the hand-built ex4_1_2/ex8_6_2 DAGs. No CUDA, no GPU needed.
add_executable(cuminlp_gams_test source/gams_test.cpp)
target_link_libraries(
    cuminlp_gams_test
    PRIVATE cuminlp::gams
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_gams_test PRIVATE cxx_std_20)
target_compile_definitions(
    cuminlp_gams_test
    PRIVATE "CUMINLP_GAMS_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data/gams\""
)
# nvs09_problem.hpp / autocorr_bern20_03_problem.hpp: the hand-built Problem
# builders shared with source/fixed_examples/nvs09.cu and
# source/fixed_examples/autocorr_bern20_03.cu, used here as an independent
# equivalence oracle for the discrete-driver .gms fixtures.
target_include_directories(
    cuminlp_gams_test PRIVATE "${PROJECT_SOURCE_DIR}/source/fixed_examples"
)
catch_discover_tests(cuminlp_gams_test)

# Host-only: GreedyEnumCompositionPolicy is plain templated C++ with no device
# code, so this doesn't need a CUDA device to run.
add_executable(cuminlp_composition_policy_test source/composition_policy_test.cpp)
target_link_libraries(
    cuminlp_composition_policy_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_composition_policy_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_composition_policy_test)

# Host-only: dag.hpp/Problem::validate() is deliberately device-free (see
# TEST_EXTENSION.md's "Problem::validate()" section), so §1-2's invariants
# are all testable without a CUDA device.
add_executable(cuminlp_dag_test source/dag_test.cpp)
target_link_libraries(
    cuminlp_dag_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_dag_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_dag_test)

# Host-only: cuminlp::region::decode::slot_bounds (region/decode.hpp) is a
# CUMINLP_HD function that's a no-op host function outside __CUDACC__, and
# search::Node::materialise (search/frontier.hpp) has no CUDA dependency, so
# §3a/§3b/§4a's decode invariants don't need a CUDA device either.
add_executable(cuminlp_decode_test source/decode_test.cpp)
target_link_libraries(
    cuminlp_decode_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_decode_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_decode_test)

# Host-only: the bounded frontier (design/BOUNDED_FRONTIER.md).
# search/frontier.hpp's IntervalPQueue and search/history.hpp's
# IntervalHistory have no CUDA dependency, and search/budget.hpp is
# deliberately device-free for exactly this reason -- the budget
# arithmetic, the eviction order, the refcounted history GC and the
# dropped-floor gating of the epilogue's claims are all decidable without a
# GPU. Only their wiring into search/driver.hpp is not.
add_executable(cuminlp_host_budget_test source/host_budget_test.cpp)
target_link_libraries(
    cuminlp_host_budget_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_host_budget_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_host_budget_test)

# Host-only: config/catalogue.hpp + config/problem_profile.hpp + config/
# resolve.hpp (PolicyProfile/ProblemProfile/resolve_shape/select_policy) are
# plain templated C++ over Problem<T>/FanOutSpec, with no
# CUDA type among them -- see those headers' top comments and
# design/POLICY_SELECTION.md §3. Links cuminlp::gams (not just cuminlp::lib)
# to parse the autocorr_bern20-03.gms/ex8_6_2.gms fixtures its evidence tests
# use, and includes source/ for nvs09_problem.hpp, the hand-built Problem
# shared with source/fixed_examples/nvs09.cu -- used instead of a .gms
# fixture so the nvs09 tests need no GPU and no .gms file either.
add_executable(cuminlp_policy_catalogue_test source/policy_catalogue_test.cpp)
target_link_libraries(
    cuminlp_policy_catalogue_test
    PRIVATE cuminlp::gams
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_policy_catalogue_test PRIVATE cxx_std_20)
target_compile_definitions(
    cuminlp_policy_catalogue_test
    PRIVATE "CUMINLP_GAMS_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data/gams\""
)
target_include_directories(
    cuminlp_policy_catalogue_test
    PRIVATE "${PROJECT_SOURCE_DIR}/source/fixed_examples"
)

catch_discover_tests(cuminlp_policy_catalogue_test)

# Host-only: config/run_spec.hpp (RunSpec/OverrideSet/Provenance/resolve) is
# plain templated C++ over the same config/catalogue.hpp types the resolver
# test above already builds host-only, plus backend/graph/cost.hpp, which is
# host-only for the same reason (see that header's top comment). Includes
# source/ for nvs09_problem.hpp, same as cuminlp_policy_catalogue_test.
add_executable(cuminlp_run_spec_test source/run_spec_test.cpp)
target_link_libraries(
    cuminlp_run_spec_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_run_spec_test PRIVATE cxx_std_20)
target_include_directories(
    cuminlp_run_spec_test
    PRIVATE "${PROJECT_SOURCE_DIR}/source/fixed_examples"
)

catch_discover_tests(cuminlp_run_spec_test)

# Host-only: report/observer.hpp (SearchObserver/ConsoleReporter) writes to
# std::cout directly, with no CUDA type in reach -- see that header's own
# includes. Redirects std::cout to a string for the duration of each check
# (same technique as dag_test.cpp's print_problem tests) to assert
# ConsoleReporter's exact spellings, which tools/minlp_status.py and
# BOUNDED_FRONTIER.md §7 both depend on (design/MODULE_REFACTOR.md §8).
add_executable(cuminlp_report_test source/report_test.cpp)
target_link_libraries(
    cuminlp_report_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_report_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_report_test)

# Host-only: search/cache.hpp's BackendCache names only backend/backend.hpp
# (RegionBackendFactory and friends) and region::Composition/FanOutSpec, none
# of them CUDA types, so its build/rebuild/eviction bookkeeping
# (design/TELEMETRY.md §4.4) is testable against a fake, host-only
# RegionBackendFactory<double> -- see source/cache_test.cpp's own top comment
# for how the fake simulates a device-memory cap without a GPU.
add_executable(cuminlp_cache_test source/cache_test.cpp)
target_link_libraries(
    cuminlp_cache_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_cache_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_cache_test)

# Launches backend/graph/subregion_sampler.cuh's sample_points_kernel directly (bypassing
# GraphBuilder/GraphReplay) to check Integer/Binary variables are always
# sampled on the integer lattice, even when they aren't the slot being
# cycled this iteration. Requires an actual CUDA device.
add_executable(cuminlp_sample_discrete_test source/sample_discrete_test.cu)
target_link_libraries(
    cuminlp_sample_discrete_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_sample_discrete_test PRIVATE cxx_std_20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      cuminlp_sample_discrete_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(cuminlp_sample_discrete_test)

# Exercises GraphReplay/GraphBuilder's public API end-to-end (build, set_domain,
# launch, candidate reporting): move-only + moved-from safety, ShapeMismatch on
# a mismatched set_domain() size, candidate/has_candidate/candidate_point
# self-consistency, and launch determinism (TEST_EXTENSION.md §5). Requires an
# actual CUDA device.
add_executable(cuminlp_graph_replay_test source/graph_replay_test.cu)
target_link_libraries(
    cuminlp_graph_replay_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_graph_replay_test PRIVATE cxx_std_20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      cuminlp_graph_replay_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(cuminlp_graph_replay_test)

# The backend role contract (design/MODULE_REFACTOR.md §12 stage 3): what any
# RegionBackendFactory must satisfy, stated in source/backend_contract.hpp
# against cuminlp::backend's abstract types alone, and run here against the
# CUDA-graph backend. A second backend adds a target, not a test. Requires an
# actual CUDA device.
add_executable(cuminlp_backend_contract_test source/backend_contract_test.cu)
target_link_libraries(
    cuminlp_backend_contract_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_backend_contract_test PRIVATE cxx_std_20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      cuminlp_backend_contract_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(cuminlp_backend_contract_test)

# Host-only: aggregate/partition.hpp and aggregate/policy.hpp
# (design/AGGREGATE_BOUNDING.md §5, §3.2) name region/, policy/ and model/ and
# no CUDA type, so the branch/refine split's invariants -- the products, the
# disjointness, and the "child_id = r / M" slot ordering the device epilogue
# is built around -- are all decidable without a GPU. Stage 2's exit criteria.
add_executable(cuminlp_aggregate_policy_test source/aggregate_policy_test.cpp)
target_link_libraries(
    cuminlp_aggregate_policy_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_aggregate_policy_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_aggregate_policy_test)

# Host-only: aggregate/selection.hpp and aggregate/frontier.hpp
# (design/AGGREGATE_BOUNDING.md §7) name no CUDA type. The load-bearing case is
# the equivalence sweep -- least-lower-bound must reproduce IntervalPQueue's
# pop order exactly, or the aggregate frontier is a different search and every
# backend-to-backend comparison is confounded by it. Stage 3's exit criteria.
add_executable(
    cuminlp_aggregate_frontier_test source/aggregate_frontier_test.cpp
)
target_link_libraries(
    cuminlp_aggregate_frontier_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_aggregate_frontier_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_aggregate_frontier_test)

# The aggregate backend's device side (design/AGGREGATE_BOUNDING.md §3.2, §4):
# the k-segment epilogue checked bitwise against a host reduction over the
# existing IntervalGraphReplay's per-subregion verdicts, plus enclosure against
# model/eval.hpp's real-valued oracle. Requires an actual CUDA device.
add_executable(cuminlp_aggregate_backend_test source/aggregate_backend_test.cu)
target_link_libraries(
    cuminlp_aggregate_backend_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_aggregate_backend_test PRIVATE cxx_std_20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      cuminlp_aggregate_backend_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(cuminlp_aggregate_backend_test)
