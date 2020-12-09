cmake_minimum_required(VERSION 3.10)

add_library(flat OBJECT)
target_sources(flat
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/op_type_generated.h
)
target_include_directories(flat PUBLIC ${CMAKE_CURRENT_LIST_DIR})
