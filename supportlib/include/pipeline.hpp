#pragma once
#include "cam_preview.hpp"
#include "input.hpp"

NAMESPACE_BEGIN

struct pipeline {
    engine_backend backend;
    mat4 projection_matrix;

    uvec2 preview_size;
    f32 aspect_ratio;
    cam_preview cam_preview_screen;

    shader_program shutter_program;
    bool anim_started;
    f32 anim_start_time;
    f32 anim_duration;

    u64 start_time;
    animation<f32> shutter_animation;

    pipeline();

    void pre_init(uvec2 preview_size, int* cam_width, int* cam_height);

#ifdef ANDROID
    void init_backend(ANativeWindow* texture_window, file_context* file_ctx);
#elif defined(LINUX)
    void init_backend();
#endif

    void render();
};

NAMESPACE_END