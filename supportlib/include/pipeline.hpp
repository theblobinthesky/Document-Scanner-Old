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

struct pipeline_args {
    ANativeWindow* texture_window;
    file_context* file_ctx;
    svec2 preview_size, cam_size;
    camera* cam;
    bool enable_dark_mode;
};

struct pipeline {
    engine_backend backend;
    ui_manager ui;

    mat4 projection_matrix;

    cam_preview cam_preview_screen;
    unwrapped_options_screen options_screen;

    u64 start_time, last_time;
    
    static camera* pre_init(svec2 preview_size, svec2& cam_size);

    pipeline(const pipeline_args& args);
    void render();
};

NAMESPACE_END