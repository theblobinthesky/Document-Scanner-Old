#pragma once
#include "cam_preview.hpp"
#include "platform.hpp"
#include "user_interface.hpp"

NAMESPACE_BEGIN

struct option_card {
    ui_manager* ui;
    std::vector<texture_asset_id> images;
    std::vector<std::string> titles;

    rect bounds;
    image img;
    text title;

    s32 index;

    option_card(ui_manager* ui, const std::vector<texture_asset_id>& images, const std::vector<std::string>& titles);
    
    void layout(f32 height);
    bool draw();
};

enum class stack_mode : u32 {
    STACK_HORIZONTAL, STACK_VERTICAL
};

struct scrollable_view {
    ui_manager* ui;
    const std::vector<option_card*> views;
    f32 spacing;
    stack_mode mode;

    rect bounds;
    f32 total_size;

    bool dragging;
    vec2 drag_start_pos;
    vec2 drag_start_delta;
    vec2 delta;

    scrollable_view(ui_manager* ui, const std::vector<option_card*>& views, f32 spacing, stack_mode mode) : 
        ui(ui), views(views), spacing(spacing), mode(mode) {}

    void layout(const rect& bounds) {
        this->bounds = bounds;
        delta = {};

        f32 height = bounds.size().y;

        for(option_card* card: views) {
            card->layout(height);
        }

        total_size = 1.0f * spacing;
        for(option_card* card: views) {
            total_size += spacing + card->bounds.size().x;
        }
    }

    s32 draw() {
        motion_event event = ui->backend->input.get_motion_event(bounds);
        if(event.type == motion_type::TOUCH_DOWN) {
            dragging = true;
            drag_start_pos   = event.pos;
            drag_start_delta = delta;
        }

        if(dragging && (event.type == motion_type::MOVE)) {
            delta = drag_start_delta + drag_start_pos - event.pos;
        }

        if(event.type == motion_type::TOUCH_UP) {
            dragging = false;
            drag_start_delta = delta;
        }

        delta.x = std::min(std::max(delta.x, 0.0f), total_size - 1.0f);

        vec2 transform = bounds.tl - vec2({ delta.x - spacing, 0 });

        for(option_card* card: views) {
            SCOPED_TRANSFORM(ui->backend, transform);
        
            card->draw();

            transform = transform + vec2({ card->bounds.size().x + spacing, 0 });
        }

        return 0;
    }
};

struct unwrapped_options_screen {
    ui_manager* ui;

    sdf_button back_button, next_button;
    rect scrollable_rect, categories_rect;
    text category_general, category_crop;

    option_card option_card_0;
    option_card option_card_1;
    option_card option_card_2;
    scrollable_view categories_scroll_view;
    
    rect unwrapped_rect;
    rect top_select_rect, bottom_select_rect;
    
    const texture* unwrapped_texture;

    round_checkbox top_select_checkbox, bottom_select_checkbox;
    bool top_selected;
    
    sdf_button desc_button;
    rect desc_rect;
    text desc_text;

    text select_text;
    
    animation<f32> blendin_animation;
    animation<f32> select_animation;

    bool discard_clicked, next_clicked;

    unwrapped_options_screen(ui_manager* ui, const rect& unwrapped_rect, const texture* unwrapped_texture);
    void draw_ui();
    void draw_select_ui();
    void draw_preview_ui();
    bool draw();
};

struct export_item_card {
    ui_manager* ui;

    rect icon_bounds;
    texture_asset_id icon;

    text title;
    round_checkbox checkbox;

    rect bounds;

    export_item_card(ui_manager* ui, texture_asset_id icon, const char* title);
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
    thread_pool* threads;
};

struct pipeline;

enum class screen_name {
    CAM_PREVIEW, UNWRAPPED_OPTIONS, EXPORT_OPTIONS
};

struct pipeline {
    engine_backend backend;
    ui_manager ui;

    mat4 projection_matrix;

    cam_preview cam_preview_screen;
    unwrapped_options_screen options_screen;
    export_options_screen export_screen;

    screen_name displayed_screen;

    u64 start_time, last_time;

    pipeline(pipeline_args& args);
    void init_camera_related(camera* cam, svec2 cam_size_px);

    void render();
};

NAMESPACE_END