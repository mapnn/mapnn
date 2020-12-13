cmake_minimum_required(VERSION 3.12)

add_library(net OBJECT)
target_sources(net
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/net.cpp
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/net.h
    )

if(MAPNN_BUILD_CAFFE)
    target_sources(net
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/caffe/caffe.pb.cc
                ${CMAKE_CURRENT_LIST_DIR}/caffe/caffe_model.cpp
        PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/caffe/caffe.pb.h
                ${CMAKE_CURRENT_LIST_DIR}/caffe/caffe_model.h
        )
    target_link_libraries(net PUBLIC libprotobuf)
    target_include_directories(net PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/caffe)
endif()
if(MAPNN_BUILD_ONNX)
    target_sources(net
        PRIVATE ${CMAKE_CURRENT_LIST_DIR}/onnx/onnx.pb.cc
                ${CMAKE_CURRENT_LIST_DIR}/onnx/onnx_model.cpp
        PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/onnx/onnx.pb.h
                ${CMAKE_CURRENT_LIST_DIR}/onnx/onnx_model.h
        )
    target_link_libraries(net PUBLIC libprotobuf)
    target_include_directories(net PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/onnx)
endif()
target_include_directories(net PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(net PRIVATE core)
target_link_libraries(net PRIVATE map)

