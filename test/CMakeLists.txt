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

# cuinterval.cuh's primitives are __device__-only (they wrap CUDA rounding
# intrinsics), so this suite launches one-thread kernels and checks results
# back on the host. Requires an actual CUDA device at test time.
add_executable(CuQCQPs_cuinterval_test source/cuinterval_test.cu)
target_link_libraries(
    CuQCQPs_cuinterval_test
    PRIVATE CuQCQPs::lib
    PRIVATE Catch2::Catch2WithMain
)
target_compile_features(CuQCQPs_cuinterval_test PRIVATE cxx_std_20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_target_properties(
      CuQCQPs_cuinterval_test PROPERTIES CUDA_ARCHITECTURES native
  )
endif()

catch_discover_tests(CuQCQPs_cuinterval_test)
