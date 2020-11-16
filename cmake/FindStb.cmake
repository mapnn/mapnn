cmake_minimum_required(VERSION 3.12)

include_directories(${PROJECT_SOURCE_DIR}/3rdparty/stb/)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/stb/LICENSE
    DESTINATION license RENAME LICENSE-stb.txt)
