cmake_minimum_required(VERSION 3.0)
# CMP0048 is related to letting CMake managing the package version for us
add_definitions(-Wno-sign-compare)
add_definitions(-Wno-strict-aliasing)
add_definitions(-Wno-unused-variable)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/
                    ${CMAKE_CURRENT_LIST_DIR}/source/
                  )

set(MNN_OBJECTS_TO_LINK "")
set(MNN_TARGETS "")

# Core
FILE(GLOB MNN_Core_SRC ${CMAKE_CURRENT_LIST_DIR}/source/core/*)
add_library(MNNCore OBJECT ${MNN_Core_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCore>)
list(APPEND MNN_TARGETS MNNCore)

# Math
FILE(GLOB MNN_Math_SRC ${CMAKE_CURRENT_LIST_DIR}/source/math/*)
add_library(MNNMath OBJECT ${MNN_Math_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMath>)
list(APPEND MNN_TARGETS MNNMath)

# Compute
FILE(GLOB MNN_Compute_SRC ${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/compute/*)
add_library(MNNCompute OBJECT ${MNN_Compute_SRC})

# X86_64 AVX/SSE
include(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/x86_x64/CMakeLists.txt)

# AArch32/64 Assemblies
include(${CMAKE_CURRENT_LIST_DIR}/source/backend/cpu/arm/CMakeLists.txt)

target_include_directories(MNNCompute PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(MNNCompute PUBLIC ${CMAKE_CURRENT_LIST_DIR}/source)
