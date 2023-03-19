cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_BUILD_TYPE Debug)

if(${TARGET_PLATFORM} STREQUAL Android)
    set(CMAKE_SYSTEM_NAME Android)
    set(CMAKE_SYSTEM_VERSION 26) # API level
    set(CMAKE_ANDROID_NDK "/home/erik/Android/Sdk/ndk/23.1.7779620")
    set(ABI ${CMAKE_ANDROID_ARCH_ABI})
elseif(${TARGET_PLATFORM} STREQUAL Linux)
    set(ABI x86_64)
endif()

project(docscanner)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp")

add_library(docscanner STATIC ${sources})
target_include_directories(docscanner PUBLIC "include" "dependencies")

set_target_properties(docscanner PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../../../bin/${TARGET_PLATFORM}/${ABI}")