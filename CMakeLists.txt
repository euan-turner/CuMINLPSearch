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

# Header-only core (dag, graph_replay, graph_driver, search, ...) plus
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

# FixedRosenbrockDriver::solve() is a plain (non-template) method, so unlike
# the rest of cuminlp_lib it needs an actual translation unit to compile.
add_library(fixed_rosenbrock_driver_lib source/fixed_rosenbrock_driver.cu)
add_library(cuminlp::fixed_rosenbrock_driver ALIAS fixed_rosenbrock_driver_lib)

target_link_libraries(fixed_rosenbrock_driver_lib PUBLIC cuminlp::lib)

target_compile_features(fixed_rosenbrock_driver_lib PUBLIC cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      fixed_rosenbrock_driver_lib PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

# ---- Examples ----

add_executable(fixed_rosenbrock source/main.cpp)

target_link_libraries(fixed_rosenbrock PRIVATE cuminlp::fixed_rosenbrock_driver)

target_compile_features(fixed_rosenbrock PRIVATE cxx_std_20)

# Same Rosenbrock instance/search as fixed_rosenbrock, but solved via
# GraphDriver -- the DAG/CUDA-graph-replay evaluator -- instead of the
# hand-written kernels in FixedRosenbrockDriver. Run alongside
# fixed_rosenbrock to compare the two.
add_executable(graph_rosenbrock source/graph_replay_rosenbrock_main.cu)

target_link_libraries(graph_rosenbrock PRIVATE cuminlp::lib)

target_compile_features(graph_rosenbrock PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_rosenbrock PROPERTIES CUDA_ARCHITECTURES native)
endif()

# 10-point Morse cluster energy minimisation (MINLPLib ex8_6_2), solved via
# GraphDriver with translational/rotational symmetry broken by fixing 6 of
# the 30 coordinates.
add_executable(graph_morse source/morse_cluster_energy.cu)

target_link_libraries(graph_morse PRIVATE cuminlp::lib)

target_compile_features(graph_morse PRIVATE cxx_std_20)



if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_morse PROPERTIES CUDA_ARCHITECTURES native)
endif()

# power series problem (MINLPLib ex4_1_2)
add_executable(graph_power_series source/power_series.cu)

target_link_libraries(graph_power_series PRIVATE cuminlp::lib)

target_compile_features(graph_power_series PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(graph_power_series PROPERTIES CUDA_ARCHITECTURES native)
endif()

add_executable(cuminlp_graph_replay_example source/graph_replay_example.cu)

target_link_libraries(cuminlp_graph_replay_example PRIVATE cuminlp::lib)

target_compile_features(cuminlp_graph_replay_example PRIVATE cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      cuminlp_graph_replay_example PROPERTIES CUDA_ARCHITECTURES native
  )
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
