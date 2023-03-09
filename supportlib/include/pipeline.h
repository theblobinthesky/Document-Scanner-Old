#pragma once
#include "android_camera.h"
#include "backend.h"
#include "assets.h"

NAMESPACE_BEGIN

struct pipeline {
    ACameraDevice* cam;
    uvec2 cam_out_size;
    ANativeWindow* texture_window;

    shader_program preview_program;
    variable proj_matrix_var;
    shader_buffer vbo;
    
    shader_program nn_input_program;
    texture nn_input_tex;

    void pre_init_preview(int* cam_width, int* cam_height);
    void init_preview(uvec2 preview_size);
    void init_cam(ANativeWindow* texture_window, docscanner::file_context* file_ctx);
    void render_preview();
};

NAMESPACE_END