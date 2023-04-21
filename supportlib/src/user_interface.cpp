#include "user_interface.hpp"
#include "stb_image.h"
#include "stb_truetype.h"

using namespace docscanner;

font_instance::font_instance(engine_backend* backend, const std::string& path, f32 height) {
    stbtt_fontinfo f = {};

    u8* data;
    u32 size;
    file_to_buffer(backend->assets->ctx, path.c_str(), data, size);
    stbtt_InitFont(&f, data, stbtt_GetFontOffsetForIndex(data, 0));


    s32 x = 1, y = 1, bottom_y = 1;
    f32 pixel_height = (height / backend->preview_height) * backend->preview_size_px.y;
    f32 scale = stbtt_ScaleForPixelHeight(&f, pixel_height);

    f32 inv_pixel_size = height / pixel_height;
   

    s32 ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&f, &ascent, &descent, &line_gap);
    font_height = scale * inv_pixel_size * (ascent - descent);
    line_height = scale * inv_pixel_size * (ascent - descent + line_gap);


    std::string chars_to_render = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 .?!:";

    atlas_size = { 1024, 1024 };
    atlas = new u8[atlas_size.area()];

    for (s32 i = 0; i < chars_to_render.size(); i++) {
        int advance, lsb, x0,y0,x1,y1;
        char c = chars_to_render.at(i);
        int g = stbtt_FindGlyphIndex(&f, c);
        stbtt_GetGlyphHMetrics(&f, g, &advance, &lsb);
        stbtt_GetGlyphBitmapBox(&f, g, scale, scale, &x0, &y0, &x1, &y1);
        
        s32 gw = x1 - x0;
        s32 gh = y1 - y0;

        if (x + gw + 1 >= atlas_size.x)
            y = bottom_y, x = 1; // advance to next row
        if (y + gh + 1 >= atlas_size.y) { // check if it fits vertically AFTER potentially moving to next row
            LOGE_AND_BREAK("Font bitmap atlas too small.");
            return;
        }

        ASSERT(x + gw < atlas_size.x, ".");
        ASSERT(y + gh < atlas_size.y, ".");

        stbtt_MakeGlyphBitmap(&f, atlas + y * atlas_size.x + x, gw, gh, atlas_size.x, scale, scale, g);

        vec2 inv_atlas_size = { 1.0f / (f32)(atlas_size.x - 1), 1.0f / (f32)(atlas_size.y - 1) };

        glyph gl = {
            .i = i,
            .uv = { 
                { x * inv_atlas_size.x, y * inv_atlas_size.y }, 
                { (x + gw) * inv_atlas_size.x, (y + gh) * inv_atlas_size.y } 
            },
            .off = { (f32)x0 * inv_pixel_size, (f32)y0 * inv_pixel_size },
            .size = { gw * inv_pixel_size, gh * inv_pixel_size },
            .x_advance = scale * advance * inv_pixel_size
        };

        glyph_map.emplace((s32)c, gl);
        
        
        x = x + gw + 1;

        if (y + gh + 1 > bottom_y) {
            bottom_y = y + gh + 1; 
        }
    }


    kerning = new f32[chars_to_render.size() * chars_to_render.size()];

    for (s32 i = 0; i < chars_to_render.size(); i++) {
        for (s32 j = 0; j < chars_to_render.size(); j++) {
            kerning[i * chars_to_render.size() + j] = 
                scale * stbtt_GetCodepointKernAdvance(&f, chars_to_render[i], chars_to_render[j]) * inv_pixel_size;
        }
    }

    f32* f32_atlas = new f32[atlas_size.area()];
    for(s32 i = 0; i < atlas_size.area(); i++) {    
        f32_atlas[i] = (f32)atlas[i] / 255.0f;
    }

    atlas_texture = make_texture(atlas_size, GL_R32F);
    set_texture_data(atlas_texture, (u8*)f32_atlas, atlas_size);
}

const glyph* font_instance::get_glyph(s32 g) const {
    auto found = glyph_map.find(g);
    ASSERT(found != glyph_map.end(), "Glyph in atlas not found: %c.", g);

    // todo: handle dynamic updating of the glyph atlas
    return &found->second;
}

f32 font_instance::get_kerning(s32 first, s32 next) const {
    const glyph* g_first = get_glyph(first);
    const glyph* g_next = get_glyph(next);

    return kerning[g_first->i * glyph_map.size() + g_next->i];
}

void font_instance::use(s32 slot) const {
    bind_texture_to_slot(slot, atlas_texture);
}

ui_theme::ui_theme(bool enable_dark_mode) {
    background_color        = enable_dark_mode ? vec3({ 0.1f, 0.1f, 0.1f }) : vec3({ 1, 1, 1 });
    background_accent_color = enable_dark_mode ? color_from_int(0xcccccc) : color_from_int(0xfefefe);
    primary_color       = enable_dark_mode ? color_from_int(0xBB86FC) : color_from_int(0x6200EE);
    primary_dark_color  = color_from_int(0x3700B3);
    foreground_color    = enable_dark_mode ? color_from_int(0xFFFFFF) : color_from_int(0x000000);
    accept_color = color_from_int(0x45d96a);
    deny_color   = color_from_int(0xd95445);
}

ui_manager::ui_manager(engine_backend* backend, bool enable_dark_mode) : backend(backend), theme(enable_dark_mode) {
    std::string font = "font.ttf";
    small_font = get_font(font, 0.08f);
    middle_font = get_font(font, 0.12f);
}

font_instance* ui_manager::get_font(const std::string& path, f32 size) {
    u32 path_hash = std::hash<std::string>()(path);
    u64 key = (u64)path_hash << 32 | *(reinterpret_cast<u64*>(&size));

    auto found = font_map.find(key);

    if(found == font_map.end()) {
        font_instance inst(backend, path, size);
        font_map.emplace(key, inst);

        found = font_map.find(key);
    }

    return &found->second;
}

rect ui_manager::get_screen_rect() const {
    return { {}, {1, backend->preview_height} };
}

text::text(engine_backend* backend, const font_instance* font, text_alignment align, const std::string str, const vec3& color) :
    backend(backend), align(align), font(font), str(str), color(color) {
    shader = backend->compile_and_link(vert_instanced_quad_src(), frag_glyph_src(0));
    quads.init(str.size());
}

text::text(engine_backend* backend, const font_instance* font, const rect& bounds, text_alignment align, const std::string str, const vec3& color) :
    text(backend, font, align, str, color) {
    layout(bounds);
}

void text::layout(const rect& bounds) {
    this->bounds = bounds;
}

void text::set_text(const std::string str) {
    this->str = str;
}

void text::render() {
    font->use(0);

    vec2 text_size = { 0.0f, 0.0f };
    for(s32 i = 0; i < str.size(); i++) {
        const glyph* g = font->get_glyph((s32)str.at(i));
        
        if(i == str.size() - 1) text_size.x += g->size.y;
        else text_size.x += g->x_advance;

        text_size.y = std::max(text_size.y, g->size.y);
    }

    vec2 tl;
    switch(align) {
    case text_alignment::TOP_LEFT: {
        tl = bounds.tl;
    } break;
    case text_alignment::CENTER_LEFT: {
        tl = { bounds.tl.x, bounds.middle().y - text_size.y * 0.5f };
    } break;
    case text_alignment::CENTER: {
        tl = { bounds.middle() - text_size * 0.5f };
    } break;
    default: {
        LOGE_AND_BREAK("Text alignment is not supported.");
    } break;
    }

    for(s32 i = 0; i < str.size(); i++) {
        const glyph* g = font->get_glyph((s32)str.at(i));

        vec2 q_pos = vec2({ tl.x, tl.y + text_size.y }) + g->off;
        
        instanced_quad& quad = quads.quads[i];
        quad.v0 = q_pos;
        quad.v1 = q_pos + vec2({ g->size.x, 0 });
        quad.v2 = q_pos + g->size;
        quad.v3 = q_pos + vec2({ 0, g->size.y });
        quad.uv_tl = g->uv.tl;
        quad.uv_br = g->uv.br;

        tl.x += g->x_advance;
        if(i != str.size() - 1) tl.x += font->get_kerning(str.at(i), str.at(i + 1));
    }

    quads.fill();

    backend->use_program(shader);
    get_variable(shader, "color").set_vec4(color);
    quads.draw();
}

button::button(ui_manager* ui, const std::string& str, const rect& crad, vec3 color) 
    : ui(ui), crad(crad), color(color), click_color(color * 1.2f),
      content(ui->backend, ui->middle_font, bounds, text_alignment::CENTER, str, ui->theme.foreground_color),
      click_animation(ui->backend, animation_curve::EASE_IN_OUT, 0, 1, 0, 0.15, 0) {}

void button::layout(const rect& bounds) {
    this->bounds = bounds;
    content.bounds = bounds;
}

bool button::draw() {
    vec3 bg_color = vec3::lerp(color, click_color, click_animation.update());
    ui->backend->draw_rounded_colored_quad(bounds, crad, bg_color);

    content.render();

    motion_event event = ui->backend->input.get_motion_event(bounds);
    bool clicked = (event.type == motion_type::TOUCH_DOWN);
    bool released = (event.type == motion_type::TOUCH_UP);
    
    if(clicked) {
        click_animation.start_value = 0.0f;
        click_animation.end_value = 1.0f;
        click_animation.start();
    }

    if(released) {
        click_animation.start_value = click_animation.value;
        click_animation.end_value = 0.0f;
        click_animation.start();
    }

    return released;
}

line_seperator::line_seperator(ui_manager* ui, vec2 left, f32 width) : ui(ui), left(left), width(width) {}

void line_seperator::draw() {
    ui->backend->draw_rounded_colored_quad(rect::from_middle_and_size({left + vec2({width / 2.0f, 0})}, {width, ui->theme.line_seperator_height}), 
        {}, ui->theme.line_seperator_color);
}

round_checkbox::round_checkbox(ui_manager* ui, bool checked) : ui(ui), checked(checked),
    checked_icon({ read_texture_from_path(ui->backend->assets, ui->backend->threads, "checked.png"), 0, {356, 356} }) {}

void round_checkbox::layout(rect bounds) {
    this->bounds = bounds;
}

void round_checkbox::set_checked(bool checked) {
    this->checked = checked;
}

void round_checkbox::draw() {
    ui->backend->draw_rounded_colored_quad(bounds, {}, ui->theme.background_accent_color);
    
    if(checked) {
        ui->backend->draw_rounded_textured_quad(bounds, {}, checked_icon, { {}, {1, 1} });
    }
}

rect docscanner::get_texture_uvs_aligned_top(const rect& r, const svec2& tex_size) {
    vec2 size = r.size();
    
    f32 r_aspect_ratio = size.y / size.x;
    f32 tex_aspect_ratio = (f32)tex_size.y / (f32)tex_size.x;
    
    LOGI("tex_size: %d, %d", tex_size.x, tex_size.y);
    LOGI("r_aspect_ratio: %f, tex_aspect_ratio: %f", r_aspect_ratio, tex_aspect_ratio);

    f32 uv_bottom = r_aspect_ratio / tex_aspect_ratio;
    return { {}, {1.0, uv_bottom} };
}

rect docscanner::get_texture_aligned_rect(const rect& r, const svec2& size, alignment align) {
    vec2 r_size = r.size();
    f32 scaled_width = r_size.y * (size.x / (f32)size.y);

    if(align == alignment::LEFT) {
        return { r.tl, { r.tl.x + scaled_width, r.br.y } };
    } else if(align == alignment::RIGHT) {
        return { { r.br.x - scaled_width, r.br.y }, r.br };
    } else {
        LOGE_AND_BREAK("Fix this.");
        return {};
    }
}