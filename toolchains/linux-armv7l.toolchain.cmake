SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR armv7 )

SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

set(CMAKE_C_COMPILER "arm-linux-gnueabi-gcc")
set(CMAKE_CXX_COMPILER "arm-linux-gnueabi-g++")

add_definitions(-mfpu=neon)
add_definitions(-march=armv7-a)
add_definitions(-mfloat-abi=softfp)
add_definitions(-mfpu=neon-vfpv4)
