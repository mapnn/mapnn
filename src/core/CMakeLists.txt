cmake_minimum_required(VERSION 3.12)

add_library(core OBJECT)
target_sources(core   
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/graph.cpp
            ${CMAKE_CURRENT_LIST_DIR}/mem_pool.cpp
            ${CMAKE_CURRENT_LIST_DIR}/node.cpp
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/graph.h
            ${CMAKE_CURRENT_LIST_DIR}/mem_pool.h
            ${CMAKE_CURRENT_LIST_DIR}/node.h
    )
target_include_directories(core PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/)
target_link_libraries(core PUBLIC map)
target_compile_options(core PUBLIC -Ofast)
target_compile_options(core PUBLIC -ffast-math)
