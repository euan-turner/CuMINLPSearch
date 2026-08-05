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

add_executable(cuminlp_test source/cuminlp_test.cpp)
target_link_libraries(cuminlp_test PRIVATE cuminlp::fixed_rosenbrock_driver)
target_compile_features(cuminlp_test PRIVATE cxx_std_20)

add_test(NAME cuminlp_test COMMAND cuminlp_test)

# Host-only: GreedyCompositionPolicy is plain templated C++ with no device
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

# Host-only: cuminlp::decode::slot_bounds (slot_decode.hpp) is a CUMINLP_HD
# function that's a no-op host function outside __CUDACC__, and
# CompositionInterval::materialise (search.hpp) has no CUDA dependency, so
# §3a/§3b/§4a's decode invariants don't need a CUDA device either.
add_executable(cuminlp_decode_test source/decode_test.cpp)
target_link_libraries(
    cuminlp_decode_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_decode_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_decode_test)

# Host-only: the bounded frontier (design/BOUNDED_FRONTIER.md). search.hpp's
# IntervalPQueue/IntervalHistory have no CUDA dependency, and host_budget.hpp
# is deliberately device-free for exactly this reason -- the budget
# arithmetic, the eviction order, the refcounted history GC and the
# dropped-floor gating of the epilogue's claims are all decidable without a
# GPU. Only their wiring into graph_driver.cuh is not.
add_executable(cuminlp_host_budget_test source/host_budget_test.cpp)
target_link_libraries(
    cuminlp_host_budget_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_host_budget_test PRIVATE cxx_std_20)

catch_discover_tests(cuminlp_host_budget_test)

# Host-only: policy_catalogue.hpp (PolicyProfile/ProblemProfile/resolve/
# select_policy) is plain templated C++ over Problem<T>/FanOutSpec, with no
# CUDA type among them -- see that header's top comment and
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

# Checks rosenbrock_kernel/interval_rosenbrock_kernel against independent CPU
# reference implementations (see rosenbrock_test.cu). Requires an actual CUDA
# device. -frounding-math is needed for the CpuRounding oracle helpers (see
# cpu_rounding.hpp); --diag-suppress=20011 silences an unrelated nvcc warning.
add_executable(cuminlp_rosenbrock_test source/rosenbrock_test.cu)
target_link_libraries(
    cuminlp_rosenbrock_test
    PRIVATE cuminlp::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(cuminlp_rosenbrock_test PRIVATE cxx_std_20)
target_compile_options(
    cuminlp_rosenbrock_test
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-frounding-math>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20011>
)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      cuminlp_rosenbrock_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(cuminlp_rosenbrock_test)

# Launches graph_replay.cuh's sample_points_kernel directly (bypassing
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
