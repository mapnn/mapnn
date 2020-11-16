cmake_minimum_required(VERSION 3.12)

add_subdirectory(caffe)
add_subdirectory(onnx)

add_library(net STATIC)
target_sources(net   
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/net.cpp
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/net.h
    )
target_include_directories(net PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/)
target_link_libraries(net PUBLIC utils)
target_link_libraries(net PUBLIC layout)
target_link_libraries(net PUBLIC core)
target_link_libraries(net PUBLIC onnx)
target_link_libraries(net PUBLIC caffe)
