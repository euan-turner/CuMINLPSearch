# ---- Tests ----

add_executable(CuQCQPs_test source/CuQCQPs_test.cpp)
target_link_libraries(CuQCQPs_test PRIVATE CuQCQPs::lib)
target_compile_features(CuQCQPs_test PRIVATE cxx_std_20)

add_test(NAME CuQCQPs_test COMMAND CuQCQPs_test)
