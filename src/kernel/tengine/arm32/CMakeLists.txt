cmake_minimum_required(VERSION 3.12)

add_library(kernel_tengine STATIC)
target_sources(kernel_tengine PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/tengine_kernel.h)
target_sources(kernel_tengine PRIVATE 
    tengine_conv_2d_direct_3x3_dilation.cpp
    tengine_conv_2d_dw_3x3.cpp
    tengine_conv_2d_dw_dilation.cpp
    tengine_conv_2d_dw.cpp
    tengine_conv_2d_dw_k5s1.cpp
    tengine_conv_2d_dw_k5s2.cpp
    tengine_conv_2d_dw_k7s1.cpp
    tengine_conv_2d_dw_k7s2.cpp
#    tengine_conv_fast_interleave.cpp
#    tengine_conv_fast_direct.cpp
#    tengine_conv_fast_gemm.cpp
    tengine_conv_2d_wino_interleave.cpp
    tengine_conv_2d_wino.cpp
    )

target_include_directories(kernel_tengine PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/)
target_compile_options(kernel_tengine PUBLIC -Ofast)
target_compile_options(kernel_tengine PUBLIC -ffast-math)
target_link_libraries(kernel_tengine PUBLIC utils)
target_link_libraries(kernel_tengine PUBLIC kernel::ref)
target_link_libraries(kernel_tengine PUBLIC layout)
#target_link_libraries(kernel_tengine PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(kernel_tengine PUBLIC tengine)
target_compile_options(kernel_tengine PRIVATE -Wno-unused-function)

add_library(kernel::tengine ALIAS kernel_tengine)
