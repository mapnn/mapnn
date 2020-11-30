cmake_minimum_required(VERSION 3.12)

file(GLOB src_ncnn ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(kernel_ncnn STATIC)
target_sources(kernel_ncnn PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/ncnn_kernel.h)
target_sources(kernel_ncnn PRIVATE ${src_ncnn})

target_include_directories(kernel_ncnn PUBLIC  ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(kernel_ncnn PUBLIC -Ofast)
target_compile_options(kernel_ncnn PUBLIC -ffast-math)
target_link_libraries(kernel_ncnn PUBLIC utils)
target_link_libraries(kernel_ncnn PUBLIC kernel::ref)
target_link_libraries(kernel_ncnn PUBLIC layout)
#target_link_libraries(kernel_ncnn PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(kernel_ncnn PUBLIC ncnn)

add_library(kernel::ncnn ALIAS kernel_ncnn)
