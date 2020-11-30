cmake_minimum_required(VERSION 3.12)

if(MSVC) 
    return()
endif()

add_executable(layer_test)
target_sources(layer_test PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/conv_test.h
    ${CMAKE_CURRENT_LIST_DIR}/depthwise_test.h
    ${CMAKE_CURRENT_LIST_DIR}/groupconv_test.h
    ${CMAKE_CURRENT_LIST_DIR}/test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/test.h
    )
target_link_libraries(layer_test PUBLIC layout)
target_link_libraries(layer_test PUBLIC core)
install(TARGETS layer_test RUNTIME DESTINATION bin)
