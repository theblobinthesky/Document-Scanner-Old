#include <jni.h>
#include "android_jni.hpp"
#include <android/native_window_jni.h>
#include <android/asset_manager_jni.h>

#define DECL_FUNC(return_type, class_name, func_name) \
    extern "C" JNIEXPORT return_type JNICALL Java_com_erikstern_documentscanner_##class_name##_##func_name

constexpr s32 MOTION_EVENT_ACTION_DOWN = 0;

DECL_FUNC(void, GLSurfaceRenderer, nativeCreate)(JNIEnv *env, jobject obj) {
    docscanner::create_persistent_pipeline(env, obj);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeDestroy)(JNIEnv *env, jobject obj) {
    docscanner::destroy_persistent_pipeline(env, obj);
}

jintArray jintArray_from_ptr(JNIEnv *env, const int *ptr, int len) {
    jintArray intJavaArray = env->NewIntArray(len);
    env->SetIntArrayRegion(intJavaArray, 0, len, ptr);

    return intJavaArray;
}


DECL_FUNC(jintArray, GLSurfaceRenderer, nativePreInit)(JNIEnv* env, jobject obj, jint preview_width, jint preview_height) {
    docscanner::pipeline* pipeline = docscanner::get_persistent_pipeline(env, obj);
    if (!pipeline) return {};

    int dimens[2];
    pipeline->pre_init({(u32) preview_width, (u32) preview_height}, dimens + 0, dimens + 1);
    return jintArray_from_ptr(env, dimens, 2);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeInit)(JNIEnv *env, jobject obj, jobject asset_mngr, jobject surface, jobject window_obj) {
    auto* mngr_from_java = AAssetManager_fromJava(env, asset_mngr);
    auto file_ctx = docscanner::get_file_ctx_from_asset_mngr(mngr_from_java);

    ANativeWindow *window = ANativeWindow_fromSurface(env, surface);

    docscanner::pipeline *pipeline = docscanner::get_persistent_pipeline(env, obj);
    if (pipeline) pipeline->init_backend(window, &file_ctx);

    /*jclass windowClass = env->FindClass("android/view/Window");
    jmethodID setStatusBarColorMethod = env->GetMethodID(windowClass, "setStatusBarColor", "(I)V");
    env->CallVoidMethod(window_obj, setStatusBarColorMethod, 0xcecece);*/
}

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

DECL_FUNC(void, GLSurfaceRenderer, nativeMotionEvent)(JNIEnv* env, jobject obj, jint event, jfloat x, jfloat y) {
    docscanner::motion_event motion_event;

    switch(event) {
        case MOTION_EVENT_ACTION_DOWN: {
            motion_event.type = docscanner::motion_type::TOUCH_DOWN;
            motion_event.pos = { x, y };
        } break;
    }

    docscanner::pipeline *pipeline = docscanner::get_persistent_pipeline(env, obj);
    if (pipeline) pipeline->backend.input.handle_motion_event(motion_event);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeRender)(JNIEnv *env, jobject obj) {
    docscanner::pipeline *pipeline = docscanner::get_persistent_pipeline(env, obj);
    if (pipeline) pipeline->render();
}