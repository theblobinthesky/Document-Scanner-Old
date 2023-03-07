#include "android_jni.h"

extern jstring test(JNIEnv* env) {
    std::string hello = "Hello from C++ changed 3";
    return env->NewStringUTF(hello.c_str());
}