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

add_executable(CuQCQPs_test source/CuQCQPs_test.cpp)
target_link_libraries(CuQCQPs_test PRIVATE CuQCQPs::lib)
target_compile_features(CuQCQPs_test PRIVATE cxx_std_20)

add_test(NAME CuQCQPs_test COMMAND CuQCQPs_test)

# Launches one-thread device kernels and also exercises the composed interval
# ops on the host via CpuRounding (see cpu_rounding.hpp, which explains
# -frounding-math below). Requires an actual CUDA device at test time.
add_executable(CuQCQPs_cuinterval_test source/cuinterval_test.cu)
target_link_libraries(
    CuQCQPs_cuinterval_test
    PRIVATE CuQCQPs::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(CuQCQPs_cuinterval_test PRIVATE cxx_std_20)
target_compile_options(
    CuQCQPs_cuinterval_test
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-frounding-math>
    # nvcc also compiles an (unreachable, host-only-called) device stub of
    # every __host__ __device__ composed op instantiated with CpuRounding;
    # 20011 warns about that dead code, not a real bug.
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20011>
)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      CuQCQPs_cuinterval_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(CuQCQPs_cuinterval_test)

# Checks rosenbrock_kernel/interval_rosenbrock_kernel against independent CPU
# reference implementations (see rosenbrock_test.cu). Requires an actual CUDA
# device; -frounding-math/-diag-suppress as above.
add_executable(CuQCQPs_rosenbrock_test source/rosenbrock_test.cu)
target_link_libraries(
    CuQCQPs_rosenbrock_test
    PRIVATE CuQCQPs::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(CuQCQPs_rosenbrock_test PRIVATE cxx_std_20)
target_compile_options(
    CuQCQPs_rosenbrock_test
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-frounding-math>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20011>
)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      CuQCQPs_rosenbrock_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(CuQCQPs_rosenbrock_test)
