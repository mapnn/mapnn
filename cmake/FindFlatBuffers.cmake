set(FLATBUFFERS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/flatbuffers)
# Add FlatBuffers directly to our build. This defines the `flatbuffers` target.
set(FLATBUFFERS_BUILD_TESTS OFF CACHE INTERNAL "Turn off FlatBuffers test")
set(FLATBUFFERS_INSTALL OFF CACHE INTERNAL "Turn off FlatBuffers install")
add_subdirectory(${FLATBUFFERS_SRC_DIR}
                 ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
                 EXCLUDE_FROM_ALL)
list(APPEND MCNN_INCLUDE_DIRS ${FLATBUFFERS_SRC_DIR}/include)
list(APPEND MCNN_LINKER_LIBS flatbuffers)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/flatbuffers/LICENSE.txt
    DESTINATION license RENAME LICENSE-flatbuffers.txt)
