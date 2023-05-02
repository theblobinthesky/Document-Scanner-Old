#include "pipeline.hpp"
#include "backend.hpp"
#include <chrono>

using namespace docscanner;

constexpr f32 paper_aspect_ratio = 1.41421356237;

constexpr f32 cam_preview_bottom_edge = 0.1f;

constexpr f32 unwrap_top_border = 0.2f;
constexpr f32 margin = 0.05f;
constexpr f32 uw = 1.0f - 2.0f * margin;
constexpr rect unwrapped_mesh_rect = {
    { margin, uw * unwrap_top_border }, 
    { 1.0f - margin, uw * (unwrap_top_border + paper_aspect_ratio) }
};

constexpr rect desc_crad = { { 0.05f, 0 }, { 0.05f, 0 } };

constexpr vec2 min_max_button_crad = { 0.05f, 0.1f };
constexpr rect crad_thin_to_the_left = { {min_max_button_crad.x, min_max_button_crad.x}, {min_max_button_crad.y, min_max_button_crad.y} };
constexpr rect crad_thin_to_the_right = { {min_max_button_crad.y, min_max_button_crad.y}, {min_max_button_crad.x, min_max_button_crad.x} };
constexpr rect crad_even = { {min_max_button_crad.x, min_max_button_crad.x}, {min_max_button_crad.x, min_max_button_crad.x} };

constexpr f32 title_text_top = 0.15f;

void get_time(u64& start_time, f32& time) {
    auto now = std::chrono::high_resolution_clock::now();
    u64 time_long = now.time_since_epoch().count();
    
    if(start_time == 0) {
        start_time = time_long;
        time = 0.0f;
    } else {
        time = (time_long - start_time) / 1000000000.000;
    }
}

void get_split_control_button_rects(const ui_manager* ui, rect* left, rect* right) {
    rect left_right_margin = { { 0.02f, 0 }, { 0.02f, 0 } };
    f32 top = 0.8f, bottom = 0.9f;

    rect rect = ui->get_screen_rect();
    rect = get_between(rect, top, bottom);

    if(left) *left = cut_margins(grid_split(rect, 0, 2, split_direction::HORIZONTAL), left_right_margin);
    if(right) *right = cut_margins(grid_split(rect, 1, 2, split_direction::HORIZONTAL), left_right_margin);
}

void get_large_control_button_rect(const ui_manager* ui, rect& r) {
    rect left_right_margin = { { 0.02f, 0 }, { 0.02f, 0 } };
    f32 top = 0.8f, bottom = 0.9f;

    rect rect = ui->get_screen_rect();
    rect = get_between(rect, top, bottom);

    r = cut_margins(rect, left_right_margin);
}

unwrapped_options_screen::unwrapped_options_screen(ui_manager* ui, const rect& unwrapped_rect, const texture* unwrapped_texture) 
    : ui(ui), unwrapped_rect(unwrapped_rect), unwrapped_texture(unwrapped_texture),
    top_select_checkbox(ui, true), bottom_select_checkbox(ui, false), top_selected(true),
    discard_button(ui, "Discard", crad_thin_to_the_left, ui->theme.deny_color), next_button(ui, "Keep", crad_thin_to_the_right, ui->theme.accept_color),
    desc_text(ui->backend, ui->middle_font, text_alignment::CENTER, "Unenhanced", ui->theme.foreground_color),
    select_text(ui->backend, ui->middle_font, text_alignment::CENTER, "Pick an option:", ui->theme.foreground_color),
    blendin_animation(ui->backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 1.0f, 0), 
    select_animation(ui->backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 0.5f, 0) {

    rect screen_rect = {
        .tl = {},
        .br = { 1, ui->backend->preview_height }
    };

    rect screen = rect(screen_rect);

    screen = get_between(screen, 0.8f, 0.9f);

    rect discard_button_rect, next_button_rect;
    get_split_control_button_rects(ui, &discard_button_rect, &next_button_rect);
    discard_button.layout(discard_button_rect);
    next_button.layout(next_button_rect);

    desc_rect = get_at_bottom(unwrapped_rect, 0.2f );
    desc_text.layout(desc_rect);
    
    rect select_rect = cut_margins(screen_rect, { { margin, 0.25f }, { margin, 0.1f } });
    top_select_rect = cut_margins(grid_split(select_rect, 0, 2, split_direction::VERTICAL), { {0, 0}, {0, 0.015f} });
    bottom_select_rect = cut_margins(grid_split(select_rect, 1, 2, split_direction::VERTICAL), { {0, 0.015f}, {0, 0} });

    rect checkbox_rect = rect::from_middle_and_size({}, {0.1f, 0.1f});
    top_select_checkbox.layout(align_rect(cut_margins(top_select_rect, 0.05f), checkbox_rect, alignment::TOP_RIGHT));
    bottom_select_checkbox.layout(align_rect(cut_margins(bottom_select_rect, 0.05f), checkbox_rect, alignment::TOP_RIGHT));
    
    rect select_text_rect = get_at_top(select_rect, title_text_top);
    select_text.layout(select_text_rect);
}

void unwrapped_options_screen::draw_ui() {
    f32 opacity = blendin_animation.value * (1.0f - select_animation.value);
    SCOPED_COMPOSITE_GROUP(ui->backend, {}, true, opacity);

    if(discard_button.draw()) {
        LOGI("discard!");
    }

    if(next_button.draw()) {
        select_animation.start();
    }

    ui->backend->draw_rounded_colored_quad(desc_rect, desc_crad, ui->theme.background_accent_color);
    desc_text.render();
}

void unwrapped_options_screen::draw_select_ui() {
    f32 opacity = blendin_animation.value * select_animation.value;
    SCOPED_COMPOSITE_GROUP(ui->backend, {}, true, opacity);

    select_text.render();

    rect unwrapped_uv = { {}, { 1, 1 } };
    rect split_unwrapped_uv = get_texture_uvs_aligned_top(top_select_rect, unwrapped_texture->size);

    if(select_animation.state != animation_state::WAITING) {
        ui->backend->draw_rounded_textured_quad(rect::lerp(unwrapped_rect, bottom_select_rect, select_animation.value), {}, *unwrapped_texture, 
                rect::lerp(unwrapped_uv, split_unwrapped_uv, select_animation.value));

        bool top_clicked = ui->backend->input.get_motion_event(top_select_rect).type == motion_type::TOUCH_UP;
        bool bottom_clicked = ui->backend->input.get_motion_event(bottom_select_rect).type == motion_type::TOUCH_UP;
        
        if(top_clicked)     top_selected = true;
        if(bottom_clicked)  top_selected = false;

        top_select_checkbox.set_checked(top_selected);
        bottom_select_checkbox.set_checked(!top_selected);

        top_select_checkbox.draw();
        bottom_select_checkbox.draw();
    }
}

void unwrapped_options_screen::draw_preview_ui() {
    SCOPED_COMPOSITE_GROUP(ui->backend, {}, true, 1.0f);

    rect unwrapped_uv = { {}, { 1, 1 } };
    rect split_unwrapped_uv = get_texture_uvs_aligned_top(top_select_rect, unwrapped_texture->size);

    ui->backend->draw_rounded_textured_quad(rect::lerp(unwrapped_rect, top_select_rect, select_animation.value), {}, *unwrapped_texture, 
            rect::lerp(unwrapped_uv, split_unwrapped_uv, select_animation.value));
}

void unwrapped_options_screen::draw() {
    if(blendin_animation.state == animation_state::WAITING) {
        blendin_animation.start();
    }

    blendin_animation.update();
    select_animation.update();

    ui->backend->clear_screen(ui->theme.background_color);
    draw_ui();
    draw_preview_ui();
    draw_select_ui();
}

export_item_card::export_item_card(ui_manager* ui, texture_asset_id icon, const char* title) : ui(ui), icon(icon), 
    title(ui->backend, ui->small_font, text_alignment::CENTER, title, ui->theme.foreground_color), checkbox(ui, false) {}

void export_item_card::layout(rect bounds) {
    this->bounds = bounds; 

    const texture_asset* asset = ui->assets->get_texture_asset(icon);
    icon_bounds = get_texture_aligned_rect(bounds, asset->image_size, alignment::LEFT);
    
    rect title_bounds = cut_margins(bounds, { {icon_bounds.size().x, 0}, {} });
    this->title.layout(title_bounds);

    rect checkbox_bounds = get_texture_aligned_rect(bounds, asset->image_size, alignment::RIGHT);
    checkbox.layout(checkbox_bounds);
}

bool export_item_card::draw() {
    const texture_asset* asset = ui->assets->get_texture_asset(icon);
    ui->backend->draw_rounded_textured_quad(icon_bounds, {}, asset->tex, { {}, {1, 1} });

    title.render();
    checkbox.draw();

    motion_event event = ui->backend->input.get_motion_event(bounds);
    bool released = (event.type == motion_type::TOUCH_UP);

    if(released) {
        checkbox.set_checked(!checkbox.checked);
    }

    return released;
}

export_options_screen::export_options_screen(ui_manager* ui) : ui(ui),
    finish_button(ui, "Finish", crad_even, ui->theme.accept_color),
    dialogue_animation(ui->backend, animation_curve::EASE_IN_OUT, 0, 1, 0, 1.0f, 0),
    export_text(ui->backend, ui->middle_font, text_alignment::CENTER, "Please select an option:", ui->theme.foreground_color) {
    
    rect screen = ui->get_screen_rect();

    f32 card_height = 0.05f;
    f32 card_spacing = 0.01f;
    f32 card_top = 0.4f;

    // todo: fix this shitshow
    export_cards[EXPORT_CARD_ONENOTE] = new export_item_card(ui, ui->assets->load_texture_asset("one_note_icon"), "OneNote as Image");
    export_cards[EXPORT_CARD_GALLERY] = new export_item_card(ui, ui->assets->load_texture_asset("gallery_icon"), "Gallery");
    export_cards[EXPORT_CARD_PDF]     = new export_item_card(ui, ui->assets->load_texture_asset("pdf_icon"), "PDF Document");
    export_cards[EXPORT_CARD_DOCX]    = new export_item_card(ui, ui->assets->load_texture_asset("word_icon"), "Word Document");
    
    for(s32 i = 0; i < EXPORT_CARD_COUNT; i++) {
        f32 top = card_top + (card_height + card_spacing) * i;
        rect card_rect = cut_margins(get_between(screen, top, top + card_height), { {0.02f, 0}, {0.02f, 0} });
        export_cards[i]->layout(card_rect);

        if(i > 0) {
            line_seperators[i - 1] = new line_seperator(ui, card_rect.tl - vec2({0, card_spacing / 2.0f}), card_rect.size().x);
        }
    }
    

    rect finish_button_rect;
    get_large_control_button_rect(ui, finish_button_rect);
    finish_button.layout(finish_button_rect);

    dialogue_rect_small = cut_margins(screen, 0.15f);
    dialogue_rect_large = cut_margins(screen, 0.05f);

    rect export_text_rect = get_at_top(screen, 0.5f);
    export_text.layout(export_text_rect);
}

void export_options_screen::draw_ui() {
    SCOPED_COMPOSITE_GROUP(ui->backend, {}, true, 1.0f - dialogue_animation.value);

    export_text.render();

    for(s32 i = 0; i < EXPORT_CARD_COUNT; i++) {
        export_cards[i]->draw();
    }

    for(s32 i = 0; i < EXPORT_CARD_COUNT - 1; i++) {
        line_seperators[i]->draw();
    }

    finish_button.draw();
}

void export_options_screen::draw_dialogue_ui() {
    SCOPED_COMPOSITE_GROUP(ui->backend, {}, true, dialogue_animation.value);

    rect dialogue_rect = rect::lerp(dialogue_rect_small, dialogue_rect_large, dialogue_animation.value);
    ui->backend->draw_rounded_colored_quad(dialogue_rect, { {0.05f, 0.05f}, {0.05f, 0.05f} }, ui->theme.background_accent_color);
}

void export_options_screen::draw() {
    ui->backend->clear_screen(ui->theme.background_color);

    dialogue_animation.update();

    draw_ui();
    draw_dialogue_ui();
}

pipeline::pipeline(pipeline_args& args)
    : backend(args.assets->ctx, args.threads, args.preview_size), ui(&backend, args.assets, args.enable_dark_mode),
    cam_preview_screen(&backend, &ui, cam_preview_bottom_edge, unwrapped_mesh_rect), 
    options_screen(&ui, unwrapped_mesh_rect, &cam_preview_screen.tex_sampler.output_tex),
    export_screen(&ui),
    start_time(0), last_time(0) {
    projection_matrix = mat4::orthographic(0.0f, 1.0f, backend.preview_height, 0.0f, -1.0f, 1.0f);
}

void pipeline::init_camera_related(camera* cam, svec2 cam_size_px) {
    backend.init_camera_related(cam, cam_size_px);
    cam_preview_screen.init_camera_related();
}

void docscanner::pipeline::render() {
    SCOPED_CAMERA_MATRIX(&backend, projection_matrix);

    get_time(start_time, backend.time);

    bool redraw = false;

    // displayed_screen = screen_name::EXPORT_OPTIONS;
    if(displayed_screen == screen_name::CAM_PREVIEW) {
        redraw = true;

        if(cam_preview_screen.unwrap_animation.state == FINISHED) {
            displayed_screen = screen_name::UNWRAPPED_OPTIONS;
        } else {
            cam_preview_screen.render();
        }
    } else if(displayed_screen == screen_name::UNWRAPPED_OPTIONS) {
        if(cam_preview_screen.unwrap_animation.state == FINISHED) {
            redraw = true;
            options_screen.draw();
        }
    } else if(displayed_screen == screen_name::EXPORT_OPTIONS) {
        export_screen.draw();
        redraw = true;
    }

    
    backend.input.end_frame();


    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    if(redraw) {
        auto dur = end - last_time;
        LOGI("frame time: %ums, fps: %u", (u32)dur, (u32)(1000.0f / dur));
    }

    last_time = end;
}