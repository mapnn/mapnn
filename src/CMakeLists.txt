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
install(EXPORT mapnn DESTINATION lib/cmake)
install(TARGETS mapnn EXPORT mapnn 
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib
)
install(FILES 
    ${CMAKE_CURRENT_LIST_DIR}/net/net.h
    ${CMAKE_CURRENT_LIST_DIR}/layout/tensor.h
    ${CMAKE_CURRENT_LIST_DIR}/utils/type.h
    ${CMAKE_CURRENT_LIST_DIR}/utils/bctime.h
    DESTINATION include/mapnn)
