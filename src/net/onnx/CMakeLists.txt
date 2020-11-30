cmake_minimum_required(VERSION 3.12)

include_directories(${MAPNN_INCLUDE_DIRS})
add_library(onnx OBJECT)
target_sources(onnx 
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/onnx.pb.cc
            ${CMAKE_CURRENT_LIST_DIR}/onnx_model.cpp
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/onnx.pb.h
            ${CMAKE_CURRENT_LIST_DIR}/onnx_model.h
    )
target_include_directories(onnx PUBLIC ${CMAKE_CURRENT_LIST_DIR}/)
target_link_libraries(onnx PUBLIC utils)
target_link_libraries(onnx PUBLIC core)
target_link_libraries(onnx PUBLIC libprotobuf)
