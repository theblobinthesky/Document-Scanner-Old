cmake_minimum_required(VERSION 3.18.1)

set(LIBRARY_ROOT_DIRECTORY "/media/shared/Projekte/DocumentScanner/supportlib")

project("documentscanner")
add_library(documentscanner SHARED jni_interface.cpp)
target_include_directories(documentscanner PUBLIC "${LIBRARY_ROOT_DIRECTORY}/include")

set(LIBRARY_BIN_DIRECTORY "${LIBRARY_ROOT_DIRECTORY}/bin/${CMAKE_ANDROID_ARCH_ABI}")
target_link_directories(documentscanner PUBLIC ${LIBRARY_BIN_DIRECTORY})
set_target_properties(documentscanner PROPERTIES LINK_DEPENDS "${LIBRARY_BIN_DIRECTORY}/libdocscanner.a")

find_library(log-lib log)
find_library(camera-lib camera2ndk)
find_library(GLES-lib GLESv3)
find_library(EGL-lib EGL)

target_link_libraries(documentscanner docscanner)
target_link_libraries(documentscanner ${log-lib} ${camera-lib} ${GLES-lib} ${EGL-lib} android)

add_library(tflite SHARED IMPORTED)
set_target_properties(tflite PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libtensorflowlite_jni.so)
target_link_libraries(documentscanner tflite)