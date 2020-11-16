SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR arm )
SET ( CMAKE_SIMD_TYPE NEON32)

SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

add_definitions(-Ofast)
add_definitions(-ffast-math)
add_definitions(-std=c++11)
add_definitions(-mfpu=neon)
