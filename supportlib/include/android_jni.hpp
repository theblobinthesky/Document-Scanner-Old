#include <jni.h>
#include "types.h"
#include "pipeline.h"

namespace docscanner {
    bool create_persistent_pipeline(JNIEnv* env, jobject obj);
    bool destroy_persistent_pipeline(JNIEnv* env, jobject obj);
    pipeline* get_persistent_pipeline(JNIEnv* env, jobject obj);
};