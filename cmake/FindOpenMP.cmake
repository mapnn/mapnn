cmake_minimum_required(VERSION 3.12)

if(UNIX OR APPLE)
    return()
endif()

find_package(OpenMP)
list(APPEND MAPNN_LINKER_LIBS OpenMP::OpenMP_CXX)
