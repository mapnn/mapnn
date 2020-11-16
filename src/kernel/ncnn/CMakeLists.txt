cmake_minimum_required(VERSION 3.12)

if(MCNN_BUILD_WITH_SSE42)
    add_subdirectory(x86_64)
elseif(MCNN_BUILD_WITH_NEON)
    add_subdirectory(arm)
elseif(MCNN_BUILD_WITH_ASIMD)
    add_subdirectory(arm)
else()
endif()

