cmake_minimum_required(VERSION 3.18.1)

set(LIBRARY_ROOT_DIRECTORY "/media/shared/Projekte/CrossCompilationTemplate")

project("documentscanner")
add_library(documentscanner SHARED jni_interface.cpp)
target_include_directories(documentscanner PUBLIC "${LIBRARY_ROOT_DIRECTORY}/include")

set(LIBRARY_BIN_DIRECTORY "${LIBRARY_ROOT_DIRECTORY}/bin/${CMAKE_ANDROID_ARCH_ABI}")
target_link_directories(documentscanner PUBLIC ${LIBRARY_BIN_DIRECTORY})
set_target_properties(documentscanner PROPERTIES LINK_DEPENDS "${LIBRARY_BIN_DIRECTORY}/libdocscanner.a")

find_library(log-lib log)
target_link_libraries(documentscanner ${log-lib} docscanner)