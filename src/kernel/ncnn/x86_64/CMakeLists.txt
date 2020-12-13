cmake_minimum_required(VERSION 3.12)

file(GLOB src_ncnn ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(kernel_ncnn STATIC)
target_sources(kernel_ncnn PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/ncnn_kernel.h)
target_sources(kernel_ncnn PRIVATE ${src_ncnn})

target_include_directories(kernel_ncnn PUBLIC  ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(kernel_ncnn PRIVATE utils)
target_link_libraries(kernel_ncnn PRIVATE kernel::ref)
target_link_libraries(kernel_ncnn PRIVATE layout)
#target_link_libraries(kernel_ncnn PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(kernel_ncnn PRIVATE ncnn)
target_compile_options(kernel_ncnn PRIVATE -Ofast)
target_compile_options(kernel_ncnn PRIVATE -ffast-math)
target_compile_options(kernel_ncnn PRIVATE -Wno-unknown-pragmas)

add_library(kernel::ncnn ALIAS kernel_ncnn)
