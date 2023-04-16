#pragma once
#include "cam_preview.hpp"
#include "input.hpp"
#include "user_interface.hpp"

NAMESPACE_BEGIN

struct unwrapped_options_screen {
    ui_manager* ui;
    
    rect unwrapped_rect;
    rect top_unwrapped_rect, bottom_unwrapped_rect;
    
    const texture* unwrapped_texture;
    shader_program sampler_program;

    button discard_button, next_button;
    lines border_lines;
    lines corner_lines[4];
    lines split_lines;
    
    animation<f32> bg_blendin_animation, fg_blendin_animation;
    animation<f32> split_animation;

    text plain_text, enhanced_text;

    unwrapped_options_screen(ui_manager* ui, const rect& unwrapped_rect, const texture* unwrapped_texture);
    void draw();
};

typedef void(*cam_init_callback)(void*, svec2);

struct pipeline_args {
    ANativeWindow* texture_window;
    asset_manager* assets;
    svec2 preview_size;
    bool enable_dark_mode;
    void* cd;
    cam_init_callback cam_callback;
};

struct pipeline;

struct camera_loader_data {
    pipeline* pipe;
    void* data;
    cam_init_callback callback;
    ANativeWindow* texture_window;
};

struct camera_loader {
    camera_loader(pipeline* pipe, void* data, cam_init_callback callback, ANativeWindow* texture_window);
};

struct pipeline {
    thread_pool threads;
    camera_loader cam_loader;
    engine_backend backend;
    ui_manager ui;

    mat4 projection_matrix;

    cam_preview cam_preview_screen;
    unwrapped_options_screen options_screen;

    u64 start_time, last_time;

    pipeline(const pipeline_args& args);
    void init_camera_related(camera cam, svec2 cam_size_px);

    void render();
};

NAMESPACE_END