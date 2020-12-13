cmake_minimum_required(VERSION 3.12)

add_library(utils INTERFACE)
target_sources(utils INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/bctime.h
        ${CMAKE_CURRENT_LIST_DIR}/hypothesis_test.h
        ${CMAKE_CURRENT_LIST_DIR}/kernel.h
        ${CMAKE_CURRENT_LIST_DIR}/log.h
        ${CMAKE_CURRENT_LIST_DIR}/macro.h
        ${CMAKE_CURRENT_LIST_DIR}/map.h
        ${CMAKE_CURRENT_LIST_DIR}/memory.h
        ${CMAKE_CURRENT_LIST_DIR}/model.h
        ${CMAKE_CURRENT_LIST_DIR}/operator.h
        ${CMAKE_CURRENT_LIST_DIR}/ref.h
        ${CMAKE_CURRENT_LIST_DIR}/type.h
        )
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(utils INTERFACE flat)
if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    target_link_libraries(utils INTERFACE log)
endif()
