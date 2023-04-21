#include <jni.h>
#include "android_jni.hpp"
#include <android/native_window_jni.h>
#include <android/asset_manager_jni.h>
#include <android/native_activity.h>

#define DECL_FUNC(return_type, class_name, func_name) \
    extern "C" JNIEXPORT return_type JNICALL Java_com_erikstern_documentscanner_##class_name##_##func_name

enum motion_event {
    MOTION_EVENT_ACTION_DOWN = 0,
    MOTION_EVENT_ACTION_UP = 1,
    MOTION_EVENT_MOVE = 2
};

DECL_FUNC(void, GLSurfaceRenderer, nativeDestroy)(JNIEnv *env, jobject obj) {
    docscanner::destroy_persistent_pipeline(env, obj);
}

char* char_ptr_from_jstring(JNIEnv* env, jstring str) {
    const char* temp_buffer = env->GetStringUTFChars(str, nullptr);
    if (!temp_buffer) return null;

    size_t len = strlen(temp_buffer);
    char* buffer = new char[len + 1];
    memcpy(buffer, temp_buffer, len + 1);
    env->ReleaseStringUTFChars(str, temp_buffer);

    return buffer;
}

struct callback_data {
    JavaVM* java_vm;
    JNIEnv* env;
    jobject obj;
};

void cam_init_callback(void* data, svec2 cam_size) {
    auto* cd = reinterpret_cast<callback_data*>(data);

    cd->java_vm->AttachCurrentThread(&cd->env, null);

    _jclass* clazz = cd->env->GetObjectClass(cd->obj);
    _jmethodID* method = cd->env->GetMethodID(clazz, "preInitCallback", "(II)V");
    cd->env->CallVoidMethod(cd->obj, method, cam_size.x, cam_size.y);

    cd->env->DeleteGlobalRef(cd->obj);
    cd->java_vm->DetachCurrentThread();
}

DECL_FUNC(void, GLSurfaceRenderer, nativeInit)(JNIEnv *env, jobject obj, jobject asset_mngr, jobject surface, jstring internal_data_path,
        jint preview_width, jint preview_height, jboolean enable_dark_mode) {
    auto* mngr_from_java  = AAssetManager_fromJava(env, asset_mngr);
    auto assets = docscanner::get_assets_from_asset_mngr(mngr_from_java, char_ptr_from_jstring(env, internal_data_path));

    svec2 preview_size = { preview_width, preview_height };
    ANativeWindow *window = ANativeWindow_fromSurface(env, surface);

    JavaVM* java_vm;
    env->GetJavaVM(&java_vm);
    auto cd = new callback_data({ java_vm, env, env->NewGlobalRef(obj) });
    docscanner::pipeline_args args = {
            .texture_window = window, .assets = assets, .preview_size = preview_size, .enable_dark_mode = (bool)enable_dark_mode,
            .cd = cd, .cam_callback = cam_init_callback
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
        case MOTION_EVENT_ACTION_UP: {
            motion_event.type = docscanner::motion_type::TOUCH_UP;
            motion_event.pos = { x, y };
        } break;
        case MOTION_EVENT_MOVE: {
            motion_event.type = docscanner::motion_type::MOVE;
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