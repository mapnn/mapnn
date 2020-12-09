cmake_minimum_required(VERSION 3.12)

project(mapnn)
set(VERSION 0.0.1)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MAPNN_BUILD_ONNX         "build with onnx"   ON) # FIXME:
option(MAPNN_BUILD_CAFFE        "build with caffe"  ON) # FIXME:
option(MAPNN_BUILD_SHARED_LIB   "build shared lib"  OFF)

if(MAPNN_BUILD_ONNX OR MAPNN_BUILD_CAFFE)
    option(WITH_PROTOBUF "Enable libprotobuf" ON)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindStb.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindArchitecture.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindFlatBuffers.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindProtobuf.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindOpenMP.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindNCNN.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindMNN.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindTengine.cmake)
include_directories(${MAPNN_INCLUDE_DIRS})

add_subdirectory(src)
