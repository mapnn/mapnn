cmake_minimum_required(VERSION 3.12)

if(MCNN_BUILD_WITH_NEON)
    add_subdirectory(arm32)
endif()
if(MCNN_BUILD_WITH_ASIMD)
    add_subdirectory(arm64)
endif()
