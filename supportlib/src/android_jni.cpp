#include "android_jni.h"

jfieldID get_field_id(JNIEnv* env, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    return env->GetFieldID(cls, "nativeContext", "J");
}

bool docscanner::create_persistent_pipeline(JNIEnv* env, jobject obj) {
    auto* pipeline = new docscanner::pipeline();

    auto field_id = get_field_id(env, obj);
    if(!field_id) return false;

    env->SetLongField(obj, field_id, (jlong)pipeline);
    return true;
}

bool docscanner::destroy_persistent_pipeline(JNIEnv* env, jobject obj) {
    auto field_id = get_field_id(env, obj);
    if(!field_id) return false;

    auto pipeline = reinterpret_cast<docscanner::pipeline*>(env->GetLongField(obj, field_id));
    if(!pipeline) return false;
    
    delete pipeline;

    env->SetLongField(obj, field_id, 0);
    return true;
}

docscanner::pipeline* docscanner::get_persistent_pipeline(JNIEnv* env, jobject obj) {
    auto field_id = get_field_id(env, obj);
    if(!field_id) return null;
    return reinterpret_cast<docscanner::pipeline*>(env->GetLongField(obj, field_id));
}