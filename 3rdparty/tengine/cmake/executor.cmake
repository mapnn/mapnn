include_directories(executor/include executor/operator/include)

if(CONFIG_ARCH_ARM64)
    FILE(GLOB_RECURSE ARCH_LIB_CPP_SRCS executor/operator/arm64/*.cpp)
    FILE(GLOB_RECURSE TARGET_ARCH_FILES executor/operator/arm64/*.S)
    FILE(GLOB_RECURSE ARCH_COMMON_FILS executor/operator/common/arm/*.cpp)
    list(APPEND ARCH_LIB_CPP_SRCS ${ARCH_COMMON_FILS})
    include_directories(executor/operator/arm64/include)
endif()

if(CONFIG_ARCH_ARM32)
    FILE(GLOB_RECURSE ARCH_LIB_CPP_SRCS executor/operator/arm32/*.cpp)
    FILE(GLOB_RECURSE TARGET_ARCH_FILES executor/operator/arm32/*.S)
    FILE(GLOB_RECURSE ARCH_COMMON_FILS executor/operator/common/arm/*.cpp)
    list(APPEND ARCH_LIB_CPP_SRCS ${ARCH_COMMON_FILS})
    include_directories(executor/operator/arm32/include)
endif()

if(CONFIG_ARCH_ARM8_2)
    FILE(GLOB_RECURSE ARCH_LIB_CPP_SRCS_8_2 executor/operator/arm8_2/*.cpp)
    FILE(GLOB_RECURSE TARGET_ARCH_FILES_8_2 executor/operator/arm8_2/*.S)
    FILE(GLOB_RECURSE ARCH_COMMON_FILS executor/operator/common/arm/*.cpp)
    list(APPEND ARCH_LIB_CPP_SRCS_8_2 ${ARCH_COMMON_FILS})
    include_directories(executor/operator/arm8_2/include)
    list(APPEND ARCH_LIB_CPP_SRCS ${ARCH_LIB_CPP_SRCS_8_2})
    list(APPEND TARGET_ARCH_FILES ${TARGET_ARCH_FILES_8_2})
endif()
