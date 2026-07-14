# ---- Developer mode ----

# Developer mode enables targets and code paths in the CMake scripts that are
# only relevant for the developer(s) of CuQCQPs
if(PROJECT_IS_TOP_LEVEL)
  option(CuQCQPs_DEVELOPER_MODE "Enable developer mode" OFF)
endif()
