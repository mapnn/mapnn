cmake_minimum_required(VERSION 3.12)

add_subdirectory(flat)
add_subdirectory(utils)
add_subdirectory(ref)
add_subdirectory(layout)
add_subdirectory(kernel)
add_subdirectory(core)
add_subdirectory(map)
add_subdirectory(net)
add_subdirectory(tools/layer_test)
add_subdirectory(tools/layer_debug)
add_subdirectory(tools/net_test)
add_subdirectory(tools/model_test)


if(MAPNN_BUILD_SHARED_LIB)
    add_library(mapnn SHARED)
else()
    add_library(mapnn STATIC)
endif()

target_link_libraries(mapnn PRIVATE core)
target_link_libraries(mapnn PRIVATE map)
target_link_libraries(mapnn PRIVATE net)
target_include_directories(mapnn PRIVATE src)
install(EXPORT MapNNConfig DESTINATION lib/cmake)
install(TARGETS mapnn EXPORT MapNNConfig
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib
    INCLUDES DESTINATION include
)
install(FILES 
    ${CMAKE_CURRENT_LIST_DIR}/net/net.h
    ${CMAKE_CURRENT_LIST_DIR}/layout/tensor.h
    ${CMAKE_CURRENT_LIST_DIR}/utils/type.h
    ${CMAKE_CURRENT_LIST_DIR}/utils/bctime.h
    DESTINATION include/mapnn)
