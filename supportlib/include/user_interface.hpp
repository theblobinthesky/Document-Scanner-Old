#pragma once
#include "utils.hpp"
#include "backend.hpp"
#include <string>
#include <unordered_map>

NAMESPACE_BEGIN

struct glyph {
    s32 i;
    rect uv;
    vec2 off;
    vec2 size;
    f32 x_advance;
};

struct font_instance {
    std::unordered_map<s32, glyph> glyph_map;

    svec2 atlas_size;
    u8* atlas;
    texture atlas_texture;
    f32* kerning;

    f32 font_height, line_height;

    font_instance(engine_backend* backend, const std::string& path, f32 height);

    const glyph* get_glyph(s32 g) const;
    f32 get_kerning(s32 first, s32 next) const;
    void use(s32 slot) const;
};

struct ui_theme {
    vec3 black = { 0, 0, 0 };
    vec3 white = { 1, 1, 1 };
    vec3 background_color;
    vec3 background_accent_color;
    vec3 primary_color;
    vec3 primary_dark_color;
    vec3 foreground_color;
    vec3 deny_color;
    vec3 accept_color;

    rect middle_crad = { { 0.05f, 0.05f }, { 0.05f, 0.05f } };

    vec3 line_seperator_color = { 0.1f, 0.1f, 0.1f };
    f32 line_seperator_height = 0.002f;
    
    ui_theme(bool enable_dark_mode);
};

struct ui_manager {
    engine_backend* backend;
    ui_theme theme;
    font_instance* small_font;
    font_instance* middle_font;

    std::unordered_map<u64, font_instance> font_map;

    ui_manager(engine_backend* backend, bool enable_dark_mode);
    font_instance* get_font(const std::string& path, f32 size);
    rect get_screen_rect() const;
};

enum class text_alignment {
    TOP_LEFT, CENTER_LEFT, CENTER
};

struct text {
    engine_backend* backend;

    rect bounds;
    text_alignment align;

    const font_instance* font;
    std::string str;

    vec4 color;

    shader_program shader;
    instanced_quads quads;

    text(engine_backend* backend, const font_instance* font, text_alignment align, const std::string str, const vec3& color);
    text(engine_backend* backend, const font_instance* font, const rect& bounds, text_alignment align, const std::string str, const vec3& color);
    void layout(const rect& bounds);

    void set_text(const std::string str);
    void render();
};

struct button {
    ui_manager* ui;

    rect crad;
    rect bounds;
    vec3 color, click_color;

    font_instance* font;
    text content;

    animation<f32> click_animation;

    button(ui_manager* ui, const std::string& str, const rect& crad, vec3 color);
    void layout(const rect& bounds);
    bool draw();
};

struct line_seperator {
    ui_manager* ui;
    vec2 left;
    f32 width;

    line_seperator(ui_manager* ui, vec2 left, f32 width);
    void draw();
};

struct round_checkbox {
    ui_manager* ui;
    rect bounds;
    bool checked;

    texture checked_icon;

    round_checkbox(ui_manager* ui, bool checked);
    void layout(rect bounds);
    void set_checked(bool checked);

    void draw();
};

rect get_texture_uvs_aligned_top(const rect& r, const svec2& tex_size);

enum class alignment {
    LEFT, RIGHT
};

rect get_texture_aligned_rect(const rect& r, const svec2& size, alignment align);

NAMESPACE_END