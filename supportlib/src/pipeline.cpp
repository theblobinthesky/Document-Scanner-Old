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
const rect crad_even = { min_max_button_crad.x, min_max_button_crad.x, min_max_button_crad.x, min_max_button_crad.x };

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

void get_large_control_button_rect(const ui_manager* ui, rect& r) {
    rect left_right_margin = { { 0.02f, 0 }, { 0.02f, 0 } };
    f32 top = 0.8f, bottom = 0.9f;

    rect rect = ui->get_screen_rect();
    rect = get_between(rect, top, bottom);

    r = cut_margins(rect, left_right_margin);
}

option_card::option_card(ui_manager* ui, const std::vector<texture_asset_id>& images, const std::vector<std::string>& titles) :
    ui(ui), images(images), titles(titles), 
    img(ui, images[0]), title(ui->backend, { .font = ui->small_font, .str = titles[0], .color = ui->theme.white }) {}

void option_card::layout(f32 height) {
    bounds = rect::from_tl_and_size({}, {0, height});

    const vec2 image_size = img.get_image_size();
    rect image_rect = expand_rect(bounds, get_width_from_height_and_aspect_ratio(height, image_size.y / image_size.x), rect_side::RIGHT);
    img.layout(image_rect);

    const vec2 text_size = title.get_text_size();
    rect text_rect  = expand_rect(bounds, text_size.x, rect_side::RIGHT);
    title.layout(text_rect);
}

bool option_card::draw() {
    ui->backend->draw_rounded_colored_quad_desc({ 
        .bounds = bounds, .crad = { 0.05f, 0.05f, 0.05f, 0.05f }, 
        .color = vec4(ui->theme.background_accent_color, 0.5f), .border_color = ui->theme.white, .border_thickness = ui->theme.small_border_thickness 
    });

    img.draw();
    title.draw();

    return true;
}

unwrapped_options_screen::unwrapped_options_screen(ui_manager* ui, const rect& unwrapped_rect, const texture* unwrapped_texture) 
    : ui(ui), 
    back_button(ui, ui->theme.white, ui->assets->load_sdf_animation_asset("back"), rot_mode::ROT_0_DEG), 
    next_button(ui, ui->theme.white, ui->assets->load_sdf_animation_asset("back"), rot_mode::ROT_180_DEG),
    category_general(ui->backend, { .font = ui->small_font, .str = "GENERAL", .color = ui->theme.white, .underline = true }), 
    category_crop(ui->backend, { .font = ui->small_font, .str = "CROP", .color = ui->theme.white, .underline = true }),

    option_card_0(ui, { ui->assets->load_texture_asset("one_note_icon") }, { "OneNote" }),
    option_card_1(ui, { ui->assets->load_texture_asset("gallery_icon") }, { "Gallery" }),
    option_card_2(ui, { ui->assets->load_texture_asset("pdf_icon") }, { "Pdf" }),
    categories_scroll_view(ui, { &option_card_0, &option_card_1, &option_card_2 }, 0.05f, stack_mode::STACK_HORIZONTAL),

    unwrapped_rect(unwrapped_rect), unwrapped_texture(unwrapped_texture),
    top_select_checkbox(ui, true), bottom_select_checkbox(ui, false), top_selected(true),
    desc_button(ui, ui->theme.black, ui->assets->load_sdf_animation_asset("stripes")),
    desc_text(ui->backend, { .font = ui->small_font, .str = "UNENHANCED", .color = ui->theme.foreground_color }),
    select_text(ui->backend, { .font = ui->middle_font, .str = "Pick an option:", .color = ui->theme.foreground_color }),
    blendin_animation(ui->backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 1.0f, 0), 
    select_animation(ui->backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 0.5f, 0) {

    rect screen_rect = cut_margins(ui->get_screen_rect(), { {}, {0, 0.1f} });
    
    rect back_button_rect = align_rect(cut_margins(screen_rect, 0.05f), vec2({0.15f, 0.15f}), alignment::TOP_LEFT);
    rect next_button_rect = align_rect(cut_margins(screen_rect, 0.05f), vec2({0.15f, 0.15f}), alignment::TOP_RIGHT);

    back_button.layout(back_button_rect);
    next_button.layout(next_button_rect);

    desc_rect = get_at_bottom(unwrapped_rect, 0.15f);
    desc_button.layout(cut_margins(get_texture_aligned_rect(desc_rect, { 1, 1 }, alignment::LEFT), 0.02f));
    desc_text.layout(desc_rect);
    
    rect select_rect = cut_margins(screen_rect, { { margin, 0.25f }, { margin, 0.1f } });
    top_select_rect = cut_margins(grid_split(select_rect, 0, 2, split_direction::VERTICAL), { {0, 0}, {0, 0.015f} });
    bottom_select_rect = cut_margins(grid_split(select_rect, 1, 2, split_direction::VERTICAL), { {0, 0.015f}, {0, 0} });

    rect checkbox_rect = rect::from_middle_and_size({}, {0.1f, 0.1f});
    top_select_checkbox.layout(align_rect(cut_margins(top_select_rect, 0.05f), checkbox_rect, alignment::TOP_RIGHT));
    bottom_select_checkbox.layout(align_rect(cut_margins(bottom_select_rect, 0.05f), checkbox_rect, alignment::TOP_RIGHT));
    
    rect select_text_rect = get_at_top(select_rect, title_text_top);
    select_text.layout(select_text_rect);


    scrollable_rect = cut_bottom(screen_rect, 0.20f);
    categories_rect = cut_bottom(screen_rect, 0.15f);

    category_general.layout(grid_split(categories_rect, 0, 2, split_direction::HORIZONTAL));
    category_crop.layout(grid_split(categories_rect, 1, 2, split_direction::HORIZONTAL));

    categories_scroll_view.layout(scrollable_rect);
}

void unwrapped_options_screen::draw_ui() {
    f32 opacity = blendin_animation.value * (1.0f - select_animation.value);
    SCOPED_COMPOSITE_GROUP(ui->backend, {}, true, opacity);

    // motion_event event = ui->backend->input.get_motion_event(desc_rect);

    // if(event.type == motion_type::CLICKED) {
    //     select_animation.start();
    // }

    if(back_button.draw()) {
        discard_clicked = true;
    }

    if(next_button.draw()) {
        next_clicked = true;
    }

    // ui->backend->draw_rounded_colored_quad_desc(desc_rect, desc_crad, ui->theme.background_accent_color);
    // desc_button.draw();
    // desc_text.draw();


    category_general.draw();
    category_crop.draw();

    categories_scroll_view.draw();
}

void unwrapped_options_screen::draw_select_ui() {
    f32 opacity = blendin_animation.value * select_animation.value;
    SCOPED_COMPOSITE_GROUP(ui->backend, {}, true, opacity);

    select_text.draw();

    rect unwrapped_uv = { {}, { 1, 1 } };
    rect split_unwrapped_uv = get_texture_uvs_aligned_top(top_select_rect, unwrapped_texture->size);

    if(select_animation.state != animation_state::WAITING) {
        ui->backend->draw_rounded_textured_quad_desc({
            .bounds = rect::lerp(unwrapped_rect, bottom_select_rect, select_animation.value), 
            .tex = *unwrapped_texture, 
            .uv_bounds = rect::lerp(unwrapped_uv, split_unwrapped_uv, select_animation.value)
        });

        bool top_clicked = ui->backend->input.get_motion_event(top_select_rect).type == motion_type::CLICKED;
        bool bottom_clicked = ui->backend->input.get_motion_event(bottom_select_rect).type == motion_type::CLICKED;
        
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

    ui->backend->draw_rounded_textured_quad_desc({ 
        .bounds = rect::lerp(unwrapped_rect, top_select_rect, select_animation.value), 
        .tex = *unwrapped_texture, 
        .uv_bounds = rect::lerp(unwrapped_uv, split_unwrapped_uv, select_animation.value)
    });
}

bool unwrapped_options_screen::draw() {
    if(blendin_animation.state == animation_state::WAITING) {
        blendin_animation.start();
    }

    blendin_animation.update();
    select_animation.update();

    ui->backend->clear_screen(ui->theme.black);
    draw_ui();
    draw_preview_ui();
    draw_select_ui();

    return next_clicked;
}

export_item_card::export_item_card(ui_manager* ui, texture_asset_id icon, const char* title) : ui(ui), icon(icon), 
    title(ui->backend, { .font = ui->small_font, .str = title, .color = ui->theme.foreground_color }), checkbox(ui, false) {}

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
    ui->backend->draw_rounded_textured_quad_desc({ .bounds = icon_bounds, .tex = asset->tex });

    title.draw();
    checkbox.draw();

    motion_event event = ui->backend->input.get_motion_event(bounds);
    bool clicked = (event.type == motion_type::CLICKED);

    if(clicked) {
        checkbox.set_checked(!checkbox.checked);
    }

    return clicked;
}

export_options_screen::export_options_screen(ui_manager* ui) : ui(ui),
    finish_button(ui, "Finish", crad_even, ui->theme.accept_color),
    dialogue_animation(ui->backend, animation_curve::EASE_IN_OUT, 0, 1, 0, 1.0f, 0),
    export_text(ui->backend, { .font = ui->middle_font, .str = "Please select an option:", .color = ui->theme.foreground_color }) {
    
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

    export_text.draw();

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
    ui->backend->draw_rounded_colored_quad_desc({ .bounds = dialogue_rect, .crad = { 0.05f, 0.05f, 0.05f, 0.05f }, .color = ui->theme.background_accent_color });
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
            cam_preview_screen.is_visible = false;
        } else {
            cam_preview_screen.render();
        }
    } else if(displayed_screen == screen_name::UNWRAPPED_OPTIONS) {
        if(cam_preview_screen.unwrap_animation.state == FINISHED) {
            redraw = true;
            options_screen.draw();
            
            if(options_screen.discard_clicked) {
                // cam_preview_screen.reset();
                // displayed_screen = screen_name::CAM_PREVIEW;
            } else if(options_screen.next_clicked) {
                displayed_screen = screen_name::EXPORT_OPTIONS;
            }
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