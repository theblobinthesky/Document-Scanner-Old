#pragma once
#include "cam_preview.hpp"
#include "input.hpp"
#include "user_interface.hpp"

NAMESPACE_BEGIN

struct unwrapped_options_screen {
    ui_manager* ui;
    
    rect unwrapped_rect;
    button discard_button, next_button;

    unwrapped_options_screen(ui_manager* ui);
    void init(const rect& unwrapped_rect);
    void draw();
};

struct pipeline {
    engine_backend backend;
    ui_manager ui;

    mat4 projection_matrix;

    svec2 preview_size;
    f32 aspect_ratio;

    cam_preview cam_preview_screen;
    unwrapped_options_screen options_screen;

    u64 start_time;

    u64 last_time;
    
    pipeline();

    void pre_init(svec2 preview_size, int* cam_width, int* cam_height);

#ifdef ANDROID
    void init_backend(ANativeWindow* texture_window, file_context* file_ctx, bool enable_dark_mode);
#elif defined(LINUX)
    void init_backend(bool enable_dark_mode);
#endif

    void render();
};

NAMESPACE_END