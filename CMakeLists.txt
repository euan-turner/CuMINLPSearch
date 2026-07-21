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

add_library(
    cuminlp_lib
    source/cuminlp.cu
)
add_library(cuminlp::lib ALIAS cuminlp_lib)

target_include_directories(
    cuminlp_lib
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_link_libraries(cuminlp_lib PUBLIC cuinterval)

target_compile_features(cuminlp_lib PUBLIC cxx_std_20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(cuminlp_lib PROPERTIES CUDA_ARCHITECTURES native)
endif()

# ---- Declare executable ----

add_executable(cuminlp_exe source/main.cpp)
add_executable(cuminlp::exe ALIAS cuminlp_exe)

set_target_properties(cuminlp_exe PROPERTIES OUTPUT_NAME cuminlp)

target_link_libraries(cuminlp_exe PRIVATE cuminlp::lib)

target_compile_features(cuminlp_exe PRIVATE cxx_std_20)

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
