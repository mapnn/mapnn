cmake_minimum_required(VERSION 3.12)

project(mapnn)

option(WITH_PROTOBUF "Enable libprotobuf" ON)
option(HAVE_PTHREAD  "Enable libprotobuf" ON)

include(${PROJECT_SOURCE_DIR}/cmake/Utils.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindStb.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindArchitecture.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindFlatBuffers.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindProtobuf.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindOpenMP.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindNCNN.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindMNN.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindTengine.cmake)
include_directories(${MCNN_INCLUDE_DIRS})

add_subdirectory(src)
