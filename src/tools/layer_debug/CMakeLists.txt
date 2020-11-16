cmake_minimum_required(VERSION 3.12)

add_executable(layer_debug)
target_sources(layer_debug PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/conv_debug.h
    ${CMAKE_CURRENT_LIST_DIR}/depthwise_debug.h
    ${CMAKE_CURRENT_LIST_DIR}/groupconv_test.h
    ${CMAKE_CURRENT_LIST_DIR}/layer_debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/layer_debug.h
    )
target_link_libraries(layer_debug PRIVATE core)
install(TARGETS layer_debug RUNTIME DESTINATION bin)
