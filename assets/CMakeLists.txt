cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-sign-compare -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(packager)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp")

file(GLOB_RECURSE dep_sources CONFIGURE_DEPENDS "deps/src/*.cpp")
add_executable(packager ${sources} ${dep_sources})

target_include_directories(packager PUBLIC "deps/include")
# target_link_directories(packager PUBLIC "deps/lib")
# target_link_libraries(packager )

target_include_directories(packager PUBLIC "include" "deps/include")

set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/../bin")
set_target_properties(packager PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})