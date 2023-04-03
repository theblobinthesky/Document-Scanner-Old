#include <jni.h>
#include "android_jni.hpp"
#include <android/native_window_jni.h>
#include <android/asset_manager_jni.h>

#define DECL_FUNC(return_type, class_name, func_name) \
    extern "C" JNIEXPORT return_type JNICALL Java_com_erikstern_documentscanner_##class_name##_##func_name

constexpr s32 MOTION_EVENT_ACTION_DOWN = 0;

DECL_FUNC(void, GLSurfaceRenderer, nativeDestroy)(JNIEnv *env, jobject obj) {
    docscanner::destroy_persistent_pipeline(env, obj);
}

jlongArray jlongArray_from_ptr(JNIEnv *env, const s64 *ptr, int len) {
    jlongArray longJavaArray = env->NewLongArray(len);
    env->SetLongArrayRegion(longJavaArray, 0, len, ptr);

    return longJavaArray;
}


DECL_FUNC(jlongArray, GLSurfaceRenderer, nativePreInit)(JNIEnv* env, jobject obj, jint preview_width, jint preview_height) {
    svec2 cam_size{};
    docscanner::camera* cam = docscanner::pipeline::pre_init({preview_width, preview_height }, cam_size);

    s64 dimens[3] = { cam_size.x, cam_size.y, (s64)cam };
    return jlongArray_from_ptr(env, dimens, 3);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeInit)(JNIEnv *env, jobject obj, jobject asset_mngr, jobject surface, jobject window_obj,
        jint preview_width, jint preview_height, jint cam_width, jint cam_height, jlong cam_ptr, jboolean enable_dark_mode) {
    auto* mngr_from_java  = AAssetManager_fromJava(env, asset_mngr);
    auto file_ctx = docscanner::get_file_ctx_from_asset_mngr(mngr_from_java);
    svec2 preview_size = { preview_width, preview_height };
    svec2 cam_size = { cam_width, cam_height };

    ANativeWindow *window = ANativeWindow_fromSurface(env, surface);

    docscanner::pipeline_args args = {
            .texture_window = window, .file_ctx = &file_ctx, .preview_size = preview_size, .cam_size = cam_size,
            .cam = (docscanner::camera*)cam_ptr, .enable_dark_mode = (bool)enable_dark_mode
    };
    docscanner::create_persistent_pipeline(env, obj, args);

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