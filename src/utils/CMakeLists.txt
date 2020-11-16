cmake_minimum_required(VERSION 3.12)

add_library(utils INTERFACE )
target_include_directories(utils INTERFACE .)
target_link_libraries(utils INTERFACE flat)
