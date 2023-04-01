#pragma once
#include "utils.hpp"

#ifdef ANDROID
struct ACameraDevice;
struct ACaptureRequest;
struct ACameraCaptureSession;
struct ANativeWindow;
#elif defined(LINUX)
#include "backend.hpp"
#endif


NAMESPACE_BEGIN

#ifdef ANDROID
struct camera {
    ACameraDevice* device;
    ACaptureRequest* request;
    ACameraCaptureSession* session;
    void get();
};

void init_camera_capture(camera& cam, ANativeWindow* texture_window);
void resume_camera_capture(camera& cam);
void pause_camera_capture(camera& cam);
#elif defined(LINUX)
struct camera {
    int fd;
    u8* buffer;
    u32 buffer_size;
    f32* f32_buffer;

    uvec2 cam_size;
    texture cam_tex;
    
    void get();
};

void init_camera_capture(const camera& cam);
#endif

camera find_and_open_back_camera(const uvec2& min_size, uvec2& size);

NAMESPACE_END