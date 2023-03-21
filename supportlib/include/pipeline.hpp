#pragma once
#include "cam_preview.hpp"

NAMESPACE_BEGIN

struct pipeline {
    cam_preview cam_preview_screen;
    u64 start_time;
    f32 time;

    void pre_init(uvec2 preview_size, int* cam_width, int* cam_height);

#ifdef ANDROID
    void init_backend(ANativeWindow* texture_window, file_context* file_ctx);
#elif defined(LINUX)
    void init_backend();
#endif

    void render();
};

NAMESPACE_END