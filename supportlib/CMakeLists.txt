cmake_minimum_required(VERSION 3.4.1)

set(TARGET_PLATFORM Android)
set(CMAKE_BUILD_TYPE Release)

if(${TARGET_PLATFORM} STREQUAL Android)
    set(CMAKE_SYSTEM_NAME Android)
    set(CMAKE_SYSTEM_VERSION 26) # API level
    set(CMAKE_ANDROID_NDK "/home/erik/Android/Sdk/ndk/23.1.7779620")
endif()

project(docscanner)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp")

add_library(docscanner STATIC ${sources})
target_include_directories(docscanner PUBLIC "include")

if(${TARGET_PLATFORM} STREQUAL Android)
    set_target_properties(docscanner PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../../bin/${CMAKE_ANDROID_ARCH_ABI}")

    target_include_directories(docscanner PUBLIC "${CMAKE_SYSROOT}/usr/include")
endif()