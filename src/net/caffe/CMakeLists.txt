cmake_minimum_required(VERSION 3.12)

include_directories(${MAPNN_INCLUDE_DIRS})
add_library(caffe OBJECT)
target_sources(caffe   
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/caffe.pb.cc
            ${CMAKE_CURRENT_LIST_DIR}/caffe_model.cpp
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/caffe.pb.h
            ${CMAKE_CURRENT_LIST_DIR}/caffe_model.h
    )
target_include_directories(caffe PUBLIC ${CMAKE_CURRENT_LIST_DIR}/)
target_link_libraries(caffe PUBLIC utils)
target_link_libraries(caffe PUBLIC core)
target_link_libraries(caffe PUBLIC libprotobuf)
