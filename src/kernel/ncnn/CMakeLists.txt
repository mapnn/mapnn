cmake_minimum_required(VERSION 3.12)

if(MAPNN_BUILD_WITH_SSE42)
    add_subdirectory(x86_64)
elseif(MAPNN_BUILD_WITH_NEON)
    add_subdirectory(arm)
elseif(MAPNN_BUILD_WITH_ASIMD)
    add_subdirectory(arm)
else()
endif()

