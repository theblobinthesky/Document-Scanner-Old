#ifdef ANDROID

#include "camera.hpp"
#include "log.hpp"
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImage.h>
#include <string>

using namespace docscanner;

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

camera docscanner::find_and_open_back_camera(const svec2& min_size, svec2& size) {
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

    if (camera_index == -1) {
        return { true, null, null, null };
    }

    ACameraDevice* device;

    ACameraDevice_StateCallbacks device_callbacks = {
            .context = nullptr,
            .onDisconnected = onDisconnected,
            .onError = onError
    };

    std::string id = camera_ids->cameraIds[camera_index]; // todo: cleanup
    ACameraManager_openCamera(mng, id.c_str(), &device_callbacks, &device);

    ACameraManager_deleteCameraIdList(camera_ids);

    return { true, device, null, null };
}

void docscanner::init_camera_capture(camera& camera, ANativeWindow* texture_window) {
    ACameraDevice* cam = camera.device;

    // prepare request with desired template
    ACameraDevice_createCaptureRequest(cam, TEMPLATE_STILL_CAPTURE, &camera.request);

    // prepare temp_compute_output surface
    ANativeWindow_acquire(texture_window);

    // finalize capture request
    ACameraOutputTarget* texture_target = nullptr;
    ACameraOutputTarget_create(texture_window, &texture_target);
    ACaptureRequest_addTarget(camera.request, texture_target);

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
    ACameraDevice_createCaptureSession(cam, outputs, &state_callbacks, &camera.session);

    resume_camera_capture(camera);
}

void docscanner::resume_camera_capture(camera& cam) {
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
    ACameraCaptureSession_setRepeatingRequest(cam.session, &capture_callbacks, 1, &cam.request, nullptr);
}

void docscanner::pause_camera_capture(camera& cam) {
    ACameraCaptureSession_stopRepeating(cam.session);
}

void docscanner::camera::get() {}

#endif