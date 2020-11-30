cmake_minimum_required(VERSION 3.12)


if(NOT MAPNN_BUILD_WITH_SSE42 AND
        NOT MAPNN_BUILD_WITH_NEON AND
        NOT MAPNN_BUILD_WITH_ASIMD)
    return()
endif()

if( MAPNN_BUILD_WITH_NEON )
    set(CONFIG_ARCH_ARM32 ON CACHE INTERNAL GLOBAL)
    add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tengine)
endif()

if( MAPNN_BUILD_WITH_ASIMD )
    set(CONFIG_ARCH_ARM64 ON CACHE INTERNAL GLOBAL)
    add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tengine)
endif()
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/tengine/LICENSE
    DESTINATION license RENAME LICENSE-tengine.txt)
