#pragma once
#include "cam_preview.h"

NAMESPACE_BEGIN

struct pipeline {
    cam_preview cam_preview_screen;

    void pre_init(uvec2 preview_size, int* cam_width, int* cam_height);
    void init_backend(ANativeWindow* texture_window, file_context* file_ctx);
    void render();
};

NAMESPACE_END