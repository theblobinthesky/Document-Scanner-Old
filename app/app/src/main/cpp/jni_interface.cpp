#include <jni.h>
#include <string>
#include "android_jni.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_erikstern_documentscanner_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    return test(env);
}