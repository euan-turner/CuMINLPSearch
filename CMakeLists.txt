cmake_minimum_required(VERSION 3.24)

include(cmake/prelude.cmake)

project(
    cuminlp
    VERSION 0.1.0
    DESCRIPTION "CUDA solver for quadratically-constrained quadratic programs"
    LANGUAGES CXX CUDA
)

include(cmake/variables.cmake)

# ---- Dependencies ----

add_subdirectory(lib/cuinterval)

# ---- Declare library ----

# Header-only core (dag, graph_replay, search/driver, search, ...) plus
# cuinterval linkage and include dirs, shared by every driver and example
# below. Holds no compiled sources of its own -- it isn't any one driver's
# library, so it stays an interface target.
add_library(cuminlp_lib INTERFACE)
add_library(cuminlp::lib ALIAS cuminlp_lib)

target_include_directories(
    cuminlp_lib
    INTERFACE
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_link_libraries(cuminlp_lib INTERFACE cuinterval)

target_compile_features(cuminlp_lib INTERFACE cxx_std_20)

# GAMS scalar-format frontend: reads a .gms model into a dag::Problem. Pure
# host C++ with no CUDA and no external dependencies, so it is a CXX target --
# it compiles fast and its tests run on machines with no GPU.
add_library(gams_frontend source/gams/frontend.cpp)
add_library(cuminlp::gams ALIAS gams_frontend)

target_link_libraries(gams_frontend PUBLIC cuminlp::lib)

target_compile_features(gams_frontend PUBLIC cxx_std_20)

# Runs the frontend over a corpus and tallies what parsed and what did not, so
# that "which operator to add next" is answered by measurement.
add_executable(gams_report source/gams/report.cpp)

target_link_libraries(gams_report PRIVATE cuminlp::gams)

target_compile_features(gams_report PRIVATE cxx_std_20)

# Solves any parseable .gms through SearchDriver, with no per-instance code.
add_executable(gams_solve source/gams/solve.cu)

target_link_libraries(gams_solve PRIVATE cuminlp::gams PRIVATE cuminlp::lib)

target_compile_features(gams_solve PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(gams_solve PROPERTIES CUDA_ARCHITECTURES native)
endif()

# Solves any parseable .gms through the aggregate-bounding backend
# (design/AGGREGATE_BOUNDING.md). A separate binary from gams_solve, not a
# flag on it: the two backends are compared rather than merged, and the
# parameter families have nothing in common.
add_executable(aggregate_solve source/gams/aggregate_solve.cu)

target_link_libraries(aggregate_solve PRIVATE cuminlp::gams PRIVATE cuminlp::lib)

target_compile_features(aggregate_solve PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(aggregate_solve PROPERTIES CUDA_ARCHITECTURES native)
endif()

# Measures what subdivision buys, rather than using it: the interval hull over
# N subdomains against interval analysis on the parent box alone, swept over
# parent-box width (design/REFINEMENT_STUDY.md). Reuses the aggregate backend's
# graph at k = 1 with per-subregion bounds retained -- a mode no solve enables.
# Writes CSV, not the PARAMS/RESULT grammar, because a designed sweep is not a
# solver verdict.
add_executable(refinement_study source/gams/refinement_study.cu)

target_link_libraries(refinement_study PRIVATE cuminlp::gams PRIVATE cuminlp::lib)

target_compile_features(refinement_study PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(refinement_study PROPERTIES CUDA_ARCHITECTURES native)
endif()

# ---- Examples ----

# Rosenbrock instance solved via SearchDriver -- the DAG/CUDA-graph-replay
# evaluator.
add_executable(graph_rosenbrock source/fixed_examples/graph_rosenbrock.cu)

target_link_libraries(graph_rosenbrock PRIVATE cuminlp::lib)

target_compile_features(graph_rosenbrock PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_rosenbrock PROPERTIES CUDA_ARCHITECTURES native)
endif()

# 10-point Morse cluster energy minimisation (MINLPLib ex8_6_2), solved via
# SearchDriver with translational/rotational symmetry broken by fixing 6 of
# the 30 coordinates.
add_executable(graph_morse source/fixed_examples/graph_morse_cluster.cu)

target_link_libraries(graph_morse PRIVATE cuminlp::lib)

target_compile_features(graph_morse PRIVATE cxx_std_20)



if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_morse PROPERTIES CUDA_ARCHITECTURES native)
endif()

# power series problem (MINLPLib ex4_1_2)
add_executable(graph_power_series source/fixed_examples/graph_ex412.cu)

target_link_libraries(graph_power_series PRIVATE cuminlp::lib)

target_compile_features(graph_power_series PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_power_series PROPERTIES CUDA_ARCHITECTURES native)
endif()

# qps
add_executable(graph_qps source/fixed_examples/qps.cu)

target_link_libraries(graph_qps PRIVATE cuminlp::lib)

target_compile_features(graph_qps PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_qps PROPERTIES CUDA_ARCHITECTURES native)
endif()

# nvs09 (MINLPLib INLP, all-integer objective, no constraints)
add_executable(graph_nvs09 source/fixed_examples/nvs09.cu)

target_link_libraries(graph_nvs09 PRIVATE cuminlp::lib)

target_compile_features(graph_nvs09 PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_nvs09 PROPERTIES CUDA_ARCHITECTURES native)
endif()

# autocorr_bern20-03 (MINLPLib MBQCP, 20 binary variables, no integers/continuous)
add_executable(graph_autocorr_bern20_03 source/fixed_examples/autocorr_bern20_03.cu)

target_link_libraries(graph_autocorr_bern20_03 PRIVATE cuminlp::lib)

target_compile_features(graph_autocorr_bern20_03 PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_autocorr_bern20_03 PROPERTIES CUDA_ARCHITECTURES native)
endif()


# ---- Developer mode ----

if(NOT cuminlp_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of cuminlp"
  )
endif()

include(cmake/dev-mode.cmake)
