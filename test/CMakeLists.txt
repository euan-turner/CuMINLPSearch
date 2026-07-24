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
