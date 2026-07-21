# ---- Developer mode ----

# Developer mode enables targets and code paths in the CMake scripts that are
# only relevant for the developer(s) of cuminlp
if(PROJECT_IS_TOP_LEVEL)
  option(cuminlp_DEVELOPER_MODE "Enable developer mode" OFF)
endif()
