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

if(${TARGET_PLATFORM} STREQUAL Android)

add_library(docscanner STATIC ${sources})

elseif(${TARGET_PLATFORM} STREQUAL Linux)

file(GLOB_RECURSE dep_sources CONFIGURE_DEPENDS "deps/src/linux/*.c")
add_executable(docscanner ${sources} ${dep_sources})

target_include_directories(docscanner PUBLIC "deps/include/linux")
target_link_directories(docscanner PUBLIC "deps/lib/linux")
target_link_libraries(docscanner glfw3 GL tensorflowlite_c X11)

endif()

target_include_directories(docscanner PUBLIC "include" "deps/include")

string(TOUPPER ${TARGET_PLATFORM} TARGET_PLATFORM_UPPER)
string(TOLOWER ${TARGET_PLATFORM} TARGET_PLATFORM_LOWER)
target_compile_definitions(docscanner PUBLIC ${TARGET_PLATFORM_UPPER})

set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/../../../bin/${TARGET_PLATFORM_LOWER}/${ABI}")
set_target_properties(docscanner PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set_target_properties(docscanner PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})

set_target_properties(docscanner PROPERTIES POSITION_INDEPENDENT_CODE ON)