#include <jni.h>
#include "platform.hpp"
#include <android/native_window_jni.h>
#include <android/asset_manager_jni.h>
#include <android/native_activity.h>

using namespace docscanner;

#define DECL_FUNC(return_type, class_name, func_name) \
    extern "C" JNIEXPORT return_type JNICALL Java_com_erikstern_documentscanner_##class_name##_##func_name

DECL_FUNC(void, GLSurfaceRenderer, nativeDestroy)(JNIEnv *env, jobject obj) {
    platform_destroy(env, obj);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeInit)(JNIEnv *env, jobject obj, jobject asset_mngr, jobject surface, jstring internal_data_path,
        jint preview_width, jint preview_height, jboolean enable_dark_mode) {
    platform_init(env, obj, asset_mngr, surface, internal_data_path, preview_width, preview_height, enable_dark_mode);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeMotionEvent)(JNIEnv* env, jobject obj, jint event, jfloat x, jfloat y) {
    platform_motion_event(env, obj, event, x, y);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeRender)(JNIEnv *env, jobject obj) {
    platform_render(env, obj);
}



#if false
#include <jni.h>
#include <android/configuration.h>
#include <android/native_activity.h>

#define UI_MODE_NIGHT_YES 0x00000002

bool isDarkModeEnabled(ANativeActivity* activity) {
    JNIEnv* env;
    activity->vm->AttachCurrentThread(&env, NULL);

    jclass contextClass = env->GetObjectClass(activity->clazz);
    jmethodID getResources = env->GetMethodID(contextClass, "getResources", "()Landroid/content/res/Resources;");
    jobject resourcesObject = env->CallObjectMethod(activity->clazz, getResources);

    jclass resourcesClass = env->GetObjectClass(resourcesObject);
    jmethodID getConfiguration = env->GetMethodID(resourcesClass, "getConfiguration", "()Landroid/content/res/Configuration;");
    jobject configurationObject = env->CallObjectMethod(resourcesObject, getConfiguration);

    jclass configurationClass = env->GetObjectClass(configurationObject);
    jfieldID uiModeField = env->GetFieldID(configurationClass, "uiMode", "I");
    jint uiMode = env->GetIntField(configurationObject, uiModeField);

    bool isDarkMode = (uiMode & UI_MODE_NIGHT_YES) == UI_MODE_NIGHT_YES;

    activity->vm->DetachCurrentThread();

    return isDarkMode;
}
#endif
