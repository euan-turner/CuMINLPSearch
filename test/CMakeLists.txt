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
