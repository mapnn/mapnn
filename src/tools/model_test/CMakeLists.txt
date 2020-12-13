cmake_minimum_required(VERSION 3.12)

if(MSVC) 
    return()
endif()

include_directories( ../../../3rdparty/stb_image/)
add_executable(modeltest)
target_sources(modeltest PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(modeltest PRIVATE layout)
target_link_libraries(modeltest PRIVATE core)
target_link_libraries(modeltest PRIVATE net)
target_compile_options(modeltest PRIVATE -Wno-unused-but-set-variable)
target_compile_options(modeltest PRIVATE -Wno-sign-compare)
install(TARGETS modeltest RUNTIME DESTINATION bin)

