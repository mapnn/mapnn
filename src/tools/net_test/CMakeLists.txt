cmake_minimum_required(VERSION 3.12)

if(MSVC) 
    return()
endif()

add_executable(nettest)
target_sources(nettest PRIVATE main.cpp)
target_link_libraries(nettest PRIVATE core)
target_link_libraries(nettest PRIVATE net)
target_compile_options(nettest PRIVATE -Wno-unused-but-set-variable)
target_compile_options(nettest PRIVATE -Wno-sign-compare)
install(TARGETS nettest RUNTIME DESTINATION bin)

