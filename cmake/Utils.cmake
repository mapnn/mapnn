cmake_minimum_required(VERSION 3.12)

set(MAPNN_DEFINITIONS "") 
set(MAPNN_COMPILE_OPTIONS "") 
set(MAPNN_3RDPARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -pthread -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread -Wall -std=c++11")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O1")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O1")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D__DEBUG__ -D__DEBUG_GRAPH__")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__DEBUG__ -D__DEBUG_GRAPH__")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -fvisibility=hidden -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -fvisibility=hidden -s")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -Ofast -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast -ffast-math")



if(CMAKE_TOOLCHAIN_FILE)
    set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
    # get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX install CACHE PATH "Default install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


macro(mapnn_enable_map file option) 
    set(argc ${ARGC})
    if(${argc} GREATER 2)
        message(FATAL_ERROR "${file}: option is not a string!")
    endif()
    if(${option} STREQUAL "ON") 
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" name ${file})
        set(mapnn_include_file "${mapnn_include_file}#include \"${file}\"\n")
        set(mapnn_enable_name "${mapnn_enable_name}new ${name}(OPTION(true)),\n")
    elseif(${option} STREQUAL "OFF")
    else()
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" name ${file})
        set(mapnn_include_file "${mapnn_include_file}#include \"${file}\"\n")
        set(mapnn_enable_name "${mapnn_enable_name}new ${name}(OPTION(${option})),\n")
    endif()
endmacro()

macro(mapnn_config_map_end file) 
    configure_file(map_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/${file})
endmacro()

macro(mapnn_config_map_begin name) 
        set(mapnn_include_file "")
        set(mapnn_enable_name "")
        set(mapnn_class_name ${name})
endmacro()
