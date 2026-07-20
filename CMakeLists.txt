cmake_minimum_required(VERSION 3.24)

include(cmake/prelude.cmake)

project(
    CuQCQPs
    VERSION 0.1.0
    DESCRIPTION "CUDA solver for quadratically-constrained quadratic programs"
    LANGUAGES CXX CUDA
)

include(cmake/variables.cmake)

# ---- Dependencies ----

add_subdirectory(lib/cuinterval)

# ---- Declare library ----

add_library(
    CuQCQPs_lib
    source/CuQCQPs.cu
)
add_library(CuQCQPs::lib ALIAS CuQCQPs_lib)

target_include_directories(
    CuQCQPs_lib
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_link_libraries(CuQCQPs_lib PUBLIC cuinterval)

target_compile_features(CuQCQPs_lib PUBLIC cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(CuQCQPs_lib PROPERTIES CUDA_ARCHITECTURES native)
endif()

# ---- Declare executable ----

add_executable(CuQCQPs_exe source/main.cpp)
add_executable(CuQCQPs::exe ALIAS CuQCQPs_exe)

set_target_properties(CuQCQPs_exe PROPERTIES OUTPUT_NAME CuQCQPs)

target_link_libraries(CuQCQPs_exe PRIVATE CuQCQPs::lib)

target_compile_features(CuQCQPs_exe PRIVATE cxx_std_20)

# ---- Developer mode ----

if(NOT CuQCQPs_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of CuQCQPs"
  )
endif()

include(cmake/dev-mode.cmake)
