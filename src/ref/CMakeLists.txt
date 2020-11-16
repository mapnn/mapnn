cmake_minimum_required(VERSION 3.12)
file(GLOB src_ref ${CMAKE_CURRENT_LIST_DIR}/cpu/*.cpp)
file(GLOB hdr_ref ${CMAKE_CURRENT_LIST_DIR}/params/*.h)
add_library(ref OBJECT)
target_sources(ref   
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/reference.h
            ${hdr_ref}
    PRIVATE ${src_ref})
target_include_directories(ref PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/)
target_include_directories(ref PUBLIC ${CMAKE_CURRENT_LIST_DIR}/params)
target_compile_options(ref PUBLIC -Ofast)
#target_compile_options(ref PUBLIC -ffast-math)
target_link_libraries(ref PUBLIC layout)
add_library(kernel::ref ALIAS ref)
