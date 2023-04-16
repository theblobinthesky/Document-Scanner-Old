#pragma once
#include "utils.hpp"

#ifdef ANDROID
struct ACameraDevice;
struct ACaptureRequest;
struct ACameraCaptureSession;
struct ANativeWindow;
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
    f32* rot_f32_buffer;
    f32* f32_buffer;

    svec2 cam_size;
    u32 cam_tex;
    
    void get();
};

void init_camera_capture(const camera& cam);
void resume_camera_capture(camera& cam);
void pause_camera_capture(camera& cam);
#endif

camera find_and_open_back_camera(const svec2& min_size, svec2& size);

NAMESPACE_END