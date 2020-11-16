cmake_minimum_required(VERSION 3.12)

add_executable(nettest)
target_sources(nettest PRIVATE main.cpp)
target_link_libraries(nettest PRIVATE net)
install(TARGETS nettest RUNTIME DESTINATION bin)

