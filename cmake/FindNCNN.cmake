cmake_minimum_required(VERSION 3.12)

if(NOT MAPNN_BUILD_WITH_SSE42 AND
        NOT MAPNN_BUILD_WITH_NEON AND
        NOT MAPNN_BUILD_WITH_ASIMD)
    return()
endif()
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/ncnn)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/ncnn/LICENSE.txt
    DESTINATION license RENAME LICENSE-ncnn.txt)
