cmake_minimum_required(VERSION 3.12)

if(MAPNN_BUILD_WITH_NEON)
    add_subdirectory(arm32)
endif()
if(MAPNN_BUILD_WITH_ASIMD)
    add_subdirectory(arm64)
endif()
