SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR arm )
SET ( CMAKE_SIMD_TYPE NEON32)

SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

set(ANDROID_ABI armeabi-v7a)
set(ANDROID_STL c++_static)
set(ANDROID_TOOLCHAIN clang)
set(ANDROID_NATIVE_API_LEVEL android-20)
include($ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake)

add_definitions(-Ofast)
add_definitions(-ffast-math)
add_definitions(-mfpu=neon-vfpv4)
add_definitions(-mfloat-abi=softfp)

