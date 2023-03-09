#include <jni.h>
#include "android_jni.h"
#include <jni.h>
#include <android/native_window_jni.h>
#include <android/asset_manager_jni.h>

#define DECL_FUNC(return_type, class_name, func_name) \
    extern "C" JNIEXPORT return_type JNICALL Java_com_erikstern_documentscanner_##class_name##_##func_name

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

DECL_FUNC(jintArray, GLSurfaceRenderer, nativePreInit)(JNIEnv *env, jobject obj) {
    docscanner::pipeline *pipeline = docscanner::get_persistent_pipeline(env, obj);
    if (!pipeline) return {};

    int dimens[2];
    pipeline->pre_init_preview(dimens + 0, dimens + 1);
    return jintArray_from_ptr(env, dimens, 2);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeRenderInit)(JNIEnv *env, jobject obj,
                                                     jint preview_width, jint preview_height) {
    docscanner::pipeline *pipeline = docscanner::get_persistent_pipeline(env, obj);
    if (pipeline) pipeline->init_preview({(u32) preview_width, (u32) preview_height});
}

DECL_FUNC(void, GLSurfaceRenderer, nativeCamInit)(JNIEnv *env, jobject obj, jobject surface,
                                                  jobject asset_mngr) {
    auto *window = ANativeWindow_fromSurface(env, surface);

    docscanner::pipeline *pipeline = docscanner::get_persistent_pipeline(env, obj);

    auto* mngr_from_java = AAssetManager_fromJava(env, asset_mngr);
    auto file_ctx = docscanner::get_file_ctx_from_asset_mngr(mngr_from_java);

    if (pipeline) pipeline->init_cam(window, &file_ctx);
}

DECL_FUNC(void, GLSurfaceRenderer, nativeRender)(JNIEnv *env, jobject obj) {
    docscanner::pipeline *pipeline = docscanner::get_persistent_pipeline(env, obj);
    if (pipeline) pipeline->render_preview();
}