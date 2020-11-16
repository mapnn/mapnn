cmake_minimum_required(VERSION 3.12)

add_library(layout OBJECT)
target_sources(layout   
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tensor.cpp
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/tensor.h
            ${CMAKE_CURRENT_LIST_DIR}/L111W.h
            ${CMAKE_CURRENT_LIST_DIR}/L111W_s64.h
            ${CMAKE_CURRENT_LIST_DIR}/L1CHW.h
            ${CMAKE_CURRENT_LIST_DIR}/LNCHW.h
            ${CMAKE_CURRENT_LIST_DIR}/LCHW4.h
            ${CMAKE_CURRENT_LIST_DIR}/L1VAB.h
            ${CMAKE_CURRENT_LIST_DIR}/LUVA4.h
            ${CMAKE_CURRENT_LIST_DIR}/LUVAB.h
    )
target_include_directories(layout PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/)
target_link_libraries(layout PUBLIC utils)
