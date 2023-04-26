#include "platform.hpp"
#include "pipeline.hpp"

#include "camera.hpp"
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImage.h>
#include <string.h>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>

using namespace docscanner;

struct docscanner::camera {
    ACameraDevice* device;
    ACaptureRequest* request;
    ACameraCaptureSession* session;
    void get();
};

struct docscanner::file_context {
    AAssetManager* mngr;
    char* internal_data_path;
};

enum kotlin_motion_event {
    MOTION_EVENT_ACTION_DOWN = 0,
    MOTION_EVENT_ACTION_UP = 1,
    MOTION_EVENT_MOVE = 2
};

struct persistent_handle {
    thread_pool threads;
    pipeline* pipe;
};

struct camera_loader_data {
    persistent_handle* handle;
    camera* cam;
    svec2 preview_size, cam_size;

    JavaVM* java_vm;
    JNIEnv* env;
    jobject obj;
    ANativeWindow* texture_window;
};

void input_manager::init(svec2 preview_size, f32 aspect_ratio) {
    this->preview_size = preview_size;
    this->aspect_ratio = aspect_ratio;
}

void docscanner::input_manager::handle_motion_event(const motion_event& event) {
    LOGI("MOTION_EVENT: %u, %f, %f", event.type, event.pos.x, event.pos.y);
    
    this->event = {
        .type = event.type,
        .pos = { event.pos.x / (f32)(preview_size.x - 1), aspect_ratio * event.pos.y / (f32)(preview_size.y - 1) }
    };
}

motion_event input_manager::get_motion_event(const rect& bounds) {
    if(event.type != motion_type::NO_MOTION &&
        bounds.tl.x <= event.pos.x && event.pos.x <= bounds.br.x &&
        bounds.tl.y <= event.pos.y && event.pos.y <= bounds.br.y) {
        return event;
    }

    return {
        .type = motion_type::NO_MOTION,
        .pos = {}
    };
}

void input_manager::end_frame() {
    event = {
        .type = motion_type::NO_MOTION,
        .pos = {}
    };
}


void onDisconnected(void* context, ACameraDevice* device) {
    LOGE("onDisconnected");
}

void onError(void* context, ACameraDevice* device, int error) {
    LOGE("onError");
}

void onSessionActive(void* context, ACameraCaptureSession* session) {
}

void onSessionReady(void* context, ACameraCaptureSession* session) {
}

void onSessionClosed(void* context, ACameraCaptureSession* session) {
    LOGE("onSessionClosed");
}

void onCaptureFailed(void* context, ACameraCaptureSession* session, ACaptureRequest* request, ACameraCaptureFailure* failure) {
    LOGE("onCaptureFailed");
}

void onCaptureSequenceCompleted(void* context, ACameraCaptureSession* session, int sequenceId, int64_t frameNumber) {
}

void onCaptureSequenceAborted(void* context, ACameraCaptureSession* session, int sequenceId) {
    LOGE("onCaptureSequenceAborted");
}

void onCaptureCompleted(void* context, ACameraCaptureSession* session, ACaptureRequest* request, const ACameraMetadata* result) {
}

camera* docscanner::find_and_open_back_camera(const svec2& min_size, svec2& size) {
    camera* cam = new camera();
    ACameraManager* mng = ACameraManager_create();

    ACameraIdList* camera_ids = nullptr;
    ACameraManager_getCameraIdList(mng, &camera_ids);
    size_t camera_index = -1;

    for (size_t c = 0; c < camera_ids->numCameras; c++) {
        const char* id = camera_ids->cameraIds[c];

        ACameraMetadata* metadata = nullptr;
        ACameraManager_getCameraCharacteristics(mng, id, &metadata);

        int32_t tags_size = 0;
        const uint32_t* tags = nullptr;
        ACameraMetadata_getAllTags(metadata, &tags_size, &tags);

        for (int32_t t = 0; t < tags_size; t++) {
            if (tags[t] == ACAMERA_LENS_FACING) {
                ACameraMetadata_const_entry entry = {};
                ACameraMetadata_getConstEntry(metadata, tags[c], &entry);

                auto facing = (acamera_metadata_enum_android_lens_facing_t) (entry.data.u8[0]);
                if (facing != ACAMERA_LENS_FACING_FRONT) continue;

                ACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &entry);

                size_t min_index = 0, max_index = 0;
                s32 min_resolution = S32_MAX, max_resolution = 0;
                svec2 max_size;
                
                for (size_t e = 0; e < entry.count; e += 4) {
                    if (entry.data.i32[e + 3]) continue; // skip input streams
                    if (entry.data.i32[e + 0] != AIMAGE_FORMAT_YUV_420_888) continue; // skip wrong input formats
                    // todo: support raw photography for increased quality!

                    svec2 entry_size = { entry.data.i32[e + 1], entry.data.i32[e + 2] };
                    int32_t resolution = entry_size.x * entry_size.y;

                    // fallback max size
                    if (resolution > max_resolution) {
                        max_index = e;
                        max_resolution = resolution;
                        max_size = entry_size;
                    }

                    if (entry_size.x < min_size.x || entry_size.y < min_size.y) break; // skip too small formats
                    
                    // regular size
                    if (resolution < min_resolution) {
                        min_index = e;
                        min_resolution = resolution;
                        size = entry_size;
                    }
                }

                if (min_resolution == 0) {
                    // fallback to max resolution
                    min_index = max_index;
                    min_resolution = max_resolution;
                    size = max_size;
                }

                LOGI("Index %zu, Resolution of %dMP", min_index, min_resolution / (1000000));

                camera_index = c;
                break;
            }
        }

        ACameraMetadata_free(metadata);

        if (camera_index != -1) break;
    }

    if (camera_index != -1) {    
        ACameraDevice_StateCallbacks device_callbacks = {
                .context = nullptr,
                .onDisconnected = onDisconnected,
                .onError = onError
        };

        std::string id = camera_ids->cameraIds[camera_index]; // todo: cleanup
        ACameraManager_openCamera(mng, id.c_str(), &device_callbacks, &cam->device);

        ACameraManager_deleteCameraIdList(camera_ids);
    }

    return cam;
}

void init_camera_capture(camera* cam, ANativeWindow* texture_window) {
    ACameraDevice* device = cam->device;

    // prepare request with desired template
    ACameraDevice_createCaptureRequest(device, TEMPLATE_STILL_CAPTURE, &cam->request);

    // prepare temp_compute_output surface
    ANativeWindow_acquire(texture_window);

    // finalize capture request
    ACameraOutputTarget* texture_target = nullptr;
    ACameraOutputTarget_create(texture_window, &texture_target);
    ACaptureRequest_addTarget(cam->request, texture_target);

    // prepare capture session output...
    ACaptureSessionOutput* texture_output = nullptr;
    ACaptureSessionOutput_create(texture_window, &texture_output);

    // ...and container
    ACaptureSessionOutputContainer* outputs = nullptr;
    ACaptureSessionOutputContainer_create(&outputs);
    ACaptureSessionOutputContainer_add(outputs, texture_output);

    ACameraCaptureSession_stateCallbacks state_callbacks = {
        .context = nullptr,
        .onClosed = onSessionClosed,
        .onReady = onSessionReady,
        .onActive = onSessionActive
    };

    // prepare capture session
    ACameraDevice_createCaptureSession(device, outputs, &state_callbacks, &cam->session);

    resume_camera_capture(cam);
}

void docscanner::resume_camera_capture(camera* cam) {
    ACameraCaptureSession_captureCallbacks capture_callbacks = {
            .context = nullptr,
            .onCaptureStarted = nullptr,
            .onCaptureProgressed = nullptr,
            .onCaptureCompleted = onCaptureCompleted,
            .onCaptureFailed = onCaptureFailed,
            .onCaptureSequenceCompleted = onCaptureSequenceCompleted,
            .onCaptureSequenceAborted = onCaptureSequenceAborted,
            .onCaptureBufferLost = nullptr,
    };

    // start capturing continuously
    ACameraCaptureSession_setRepeatingRequest(cam->session, &capture_callbacks, 1, &cam->request, nullptr);
}

void docscanner::pause_camera_capture(const camera* cam) {
    ACameraCaptureSession_stopRepeating(cam->session);
}

void docscanner::get_camera_frame(const camera* cam) {}

void docscanner::read_from_package(file_context* ctx, const char* path, u8* &data, u32 &size) {
    AAsset* asset = AAssetManager_open(ctx->mngr, path, AASSET_MODE_BUFFER);
    ASSERT(asset != null, "AAsset open failed.");
    
    size = AAsset_getLength(asset);
    data = new u8[size];

    // todo: try AAsset_getBuffer

    int status = AAsset_read(asset, data, size);
    ASSERT(status >= 0, "AAsset read failed.");
}

std::string get_internal_path(file_context* ctx, const char* path) {
    return std::string(ctx->internal_data_path) + "/" + std::string(path);
}

void docscanner::read_from_internal_file(file_context* ctx, const char* path, u8* &data, u32 &size) {
    std::string full_path = get_internal_path(ctx, path);

    FILE* file = fopen(full_path.c_str(), "rb");
    if(!file) return;

    fseek(file, 0, SEEK_END);
    size = (u32)ftell(file);
    rewind(file);

    data = new u8[size];
    fread(data, size, 1, file);
    fclose(file);
}

void docscanner::write_to_internal_file(file_context* ctx, const char* path, u8* data, u32 size) {
    std::string full_path = get_internal_path(ctx, path);
    
    FILE* file = fopen(full_path.c_str(), "wb");
    fwrite(data, size, 1, file);
    fclose(file);
}

file_context* get_file_context(AAssetManager* mngr, char* internal_data_path) {
    file_context* ctx = new file_context({ mngr, internal_data_path });
    ctx->mngr = mngr;
    ctx->internal_data_path = internal_data_path;
    return ctx;
}

jfieldID get_field_id(JNIEnv* env, jobject obj) {
    jclass cls = env->GetObjectClass(obj);
    return env->GetFieldID(cls, "nativeContext", "J");
}

bool create_persistent_handle(JNIEnv* env, jobject obj, persistent_handle* handle, pipeline_args& args) {
    handle->pipe = new pipeline(args);

    auto field_id = get_field_id(env, obj);
    if(!field_id) return false;

    env->SetLongField(obj, field_id, (jlong)handle);
    return true;
}

bool destroy_persistent_handle(JNIEnv* env, jobject obj) {
    auto field_id = get_field_id(env, obj);
    if(!field_id) return false;

    auto handle = reinterpret_cast<persistent_handle*>(env->GetLongField(obj, field_id));
    if(!handle) return false;
    
    delete handle;

    env->SetLongField(obj, field_id, 0);

    return true;
}

persistent_handle* get_persistent_handle(JNIEnv* env, jobject obj) {
    auto field_id = get_field_id(env, obj);
    if(!field_id) return null;
    return reinterpret_cast<persistent_handle*>(env->GetLongField(obj, field_id));
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

void cam_init_callback_internal(void* data, svec2 cam_size) {
}

void finish_load_camera_internal(void* data) {
    auto loader_data = reinterpret_cast<camera_loader_data*>(data);

    loader_data->handle->pipe->init_camera_related(loader_data->cam, loader_data->cam_size);
    loader_data->handle->pipe->backend.cam_is_init = true;
}

void load_camera_internal(void* data) {
    auto loader_data = reinterpret_cast<camera_loader_data*>(data);
    loader_data->cam = find_and_open_back_camera(loader_data->preview_size, loader_data->cam_size);


    // jni code to setup the default buffer size
    JavaVM* java_vm = loader_data->java_vm;
    JNIEnv* env = loader_data->env;
    jobject obj = loader_data->obj;
    ANativeWindow* texture_window = loader_data->texture_window;

    java_vm->AttachCurrentThread(&env, null);
    _jclass* clazz = env->GetObjectClass(obj);
    
    // This calls: surfaceTexture.setDefaultBufferSize(width, height)
    jobject surface_texture = env->GetObjectField(obj, env->GetFieldID(clazz, "surfaceTexture", "Landroid/graphics/SurfaceTexture;"));
    _jclass* surface_texture_clazz = env->GetObjectClass(surface_texture);
    _jmethodID* method = env->GetMethodID(surface_texture_clazz, "setDefaultBufferSize", "(II)V");
    env->CallVoidMethod(surface_texture, method, loader_data->cam_size.x, loader_data->cam_size.y);

    env->DeleteGlobalRef(obj);
    java_vm->DetachCurrentThread();


    // initialize camera capture
    init_camera_capture(loader_data->cam, texture_window);
    resume_camera_capture(loader_data->cam);

    loader_data->handle->threads.push_gui({ finish_load_camera_internal, loader_data });
}

void docscanner::platform_init(JNIEnv *env, jobject obj, jobject asset_mngr, jobject surface, jstring internal_data_path,
        jint preview_width, jint preview_height, jboolean enable_dark_mode) {
    persistent_handle* handle = new persistent_handle();

    auto* mngr_from_java  = AAssetManager_fromJava(env, asset_mngr);
    file_context* file_ctx = new file_context({ mngr_from_java, char_ptr_from_jstring(env, internal_data_path) });
    asset_manager* assets = new asset_manager(file_ctx, "test.assetpack", &handle->threads);

    svec2 preview_size = { preview_width, preview_height };
    ANativeWindow *window = ANativeWindow_fromSurface(env, surface);

    JavaVM* java_vm;
    env->GetJavaVM(&java_vm);

    jobject global_obj = env->NewGlobalRef(obj);

    camera_loader_data* loader_data = new camera_loader_data({
        handle, null, preview_size, {},
        java_vm, env, global_obj, window
    });

    handle->threads.push({ load_camera_internal, loader_data });

    pipeline_args args = {
        .texture_window = window, .assets = assets, .preview_size = preview_size, 
        .enable_dark_mode = (bool)enable_dark_mode, .cam_callback = cam_init_callback_internal,
        .threads = &handle->threads
    };

    create_persistent_handle(env, obj, handle, args);
}

void docscanner::platform_destroy(JNIEnv* env, jobject obj) {
    destroy_persistent_handle(env, obj);
}

void docscanner::platform_motion_event(JNIEnv* env, jobject obj, jint event, jfloat x, jfloat y) {
    motion_event motion_event;

    switch(event) {
        case MOTION_EVENT_ACTION_DOWN: {
            motion_event.type = motion_type::TOUCH_DOWN;
            motion_event.pos = { x, y };
        } break;
        case MOTION_EVENT_ACTION_UP: {
            motion_event.type = motion_type::TOUCH_UP;
            motion_event.pos = { x, y };
        } break;
        case MOTION_EVENT_MOVE: {
            motion_event.type = motion_type::MOVE;
            motion_event.pos = { x, y };
        } break;
    }

    persistent_handle *handle = get_persistent_handle(env, obj);
    if (handle) handle->pipe->backend.input.handle_motion_event(motion_event);
}

void docscanner::platform_render(JNIEnv *env, jobject obj) {
    persistent_handle *handle = get_persistent_handle(env, obj);
    handle->threads.work_on_gui_queue();

    if (handle) handle->pipe->render();
}