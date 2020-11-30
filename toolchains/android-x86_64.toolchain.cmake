SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR x86_64 )

SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

set(ANDROID_ABI x86_64)
set(ANDROID_STL c++_static)
set(ANDROID_NATIVE_API_LEVEL android-20)
set(MAPNN_LINKER_LIBS log)
if(EXISTS $ENV{ANDROID_HOME}/build/cmake/android.toolchain.cmake)
    include($ENV{ANDROID_HOME}/build/cmake/android.toolchain.cmake)
elseif(EXISTS $ENV{ANDROID_HOME}/ndk-bundle/build/cmake/android.toolchain.cmake)
    include($ENV{ANDROID_HOME}/ndk-bundle/build/cmake/android.toolchain.cmake)
else()
    message(FATAL_ERROR "Filed to find ANDROID_HOME")
endif()
