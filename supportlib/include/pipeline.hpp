#pragma once
#include "cam_preview.hpp"
#include "input.hpp"
#include "user_interface.hpp"

NAMESPACE_BEGIN

struct unwrapped_options_screen {
    ui_manager* ui;
    
    rect unwrapped_rect;
    rect top_select_rect, bottom_select_rect;
    
    const texture* unwrapped_texture;

    button discard_button, next_button;
    lines border_lines;
    lines corner_lines[4];
    lines split_lines;

    rect desc_rect;
    text desc_text;

    text select_text;
    
    animation<f32> blendin_animation;
    animation<f32> select_animation;

    unwrapped_options_screen(ui_manager* ui, const rect& unwrapped_rect, const texture* unwrapped_texture);
    void draw_ui();
    void draw_select_ui();
    void draw_preview_ui();
    void draw();
};

struct export_item_card {
    ui_manager* ui;

    rect icon_bounds;
    texture icon;

    text title;
    round_checkbox checkbox;

    rect bounds;

    export_item_card(ui_manager* ui, texture icon, const char* title);
    void layout(rect bounds);
    bool draw();
};

enum export_cards {
    EXPORT_CARD_ONENOTE,
    EXPORT_CARD_GALLERY,
    EXPORT_CARD_PDF,
    EXPORT_CARD_DOCX,
    EXPORT_CARD_COUNT
};

struct export_options_screen {
    ui_manager* ui;

    export_item_card* export_cards[EXPORT_CARD_COUNT];
    line_seperator* line_seperators[EXPORT_CARD_COUNT - 1];

    button finish_button;

    rect dialogue_rect_small, dialogue_rect_large;

    animation<f32> dialogue_animation;

    text export_text;

    export_options_screen(ui_manager* ui);
    void draw_ui();
    void draw_dialogue_ui();
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

enum class screen_name {
    CAM_PREVIEW, UNWRAPPED_OPTIONS, EXPORT_OPTIONS
};

struct pipeline {
    thread_pool threads;
    camera_loader cam_loader;
    engine_backend backend;
    ui_manager ui;

    mat4 projection_matrix;

    cam_preview cam_preview_screen;
    unwrapped_options_screen options_screen;
    export_options_screen export_screen;

    screen_name displayed_screen;

    u64 start_time, last_time;

    pipeline(const pipeline_args& args);
    void init_camera_related(camera cam, svec2 cam_size_px);

    void render();
};

NAMESPACE_END