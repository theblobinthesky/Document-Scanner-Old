#ifdef ANDROID

#include <jni.h>
#include "utils.hpp"
#include "pipeline.hpp"

namespace docscanner {
    bool create_persistent_pipeline(JNIEnv* env, jobject obj, const pipeline_args& args);
    bool destroy_persistent_pipeline(JNIEnv* env, jobject obj);
    pipeline* get_persistent_pipeline(JNIEnv* env, jobject obj);
};

#endif