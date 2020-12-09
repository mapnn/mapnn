cmake_minimum_required(VERSION 3.12)

add_library(utils OBJECT)
target_sources(utils PUBLIC  
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
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(utils PUBLIC flat)
