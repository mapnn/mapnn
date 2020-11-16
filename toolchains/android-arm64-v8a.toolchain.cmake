SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR armv7 )

SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

set(ANDROID_ABI arm64-v8a)
set(ANDROID_STL c++_static)
set(ANDROID_TOOLCHAIN clang)
set(ANDROID_PLATFORM android-21)
include($ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake)

add_definitions(-Ofast)
add_definitions(-ffast-math)

