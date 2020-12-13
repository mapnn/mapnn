cmake_minimum_required(VERSION 3.10)

add_library(flat INTERFACE)
target_sources(flat
    INTERFACE ${CMAKE_CURRENT_LIST_DIR}/op_type_generated.h
)
target_include_directories(flat INTERFACE ${CMAKE_CURRENT_LIST_DIR})
