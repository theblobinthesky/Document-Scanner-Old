#pragma once
#include "utils.hpp"

#ifdef ANDROID
struct ACameraDevice;
struct ANativeWindow;
#elif defined(LINUX)
#include "backend.hpp"
#endif


NAMESPACE_BEGIN

#ifdef ANDROID
struct camera {
    ACameraDevice* device;
    void get();
};

void init_camera_capture(const camera& cam, ANativeWindow* texture_window);
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