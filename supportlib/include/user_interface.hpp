#pragma once
#include "utils.hpp"
#include "backend.hpp"
#include <string>
#include <unordered_map>

NAMESPACE_BEGIN

struct font_family;

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
    vec3 background_color;
    vec3 primary_color;
    vec3 primary_dark_color;
    
    ui_theme(bool enable_dark_mode);
};

struct ui_manager {
    engine_backend* backend;
    ui_theme theme;

    std::unordered_map<u64, font_instance> font_map;

    ui_manager(engine_backend* backend, bool enable_dark_mode);
    font_instance* get_font(const std::string& path, f32 size);
};

enum class text_alignment {
    TOP_LEFT, CENTER
};

struct text {
    engine_backend* backend;

    rect bounds;
    text_alignment align;

    const font_instance* font;
    std::string str;

    shader_program shader;
    instanced_quads quads;

    text(engine_backend* backend, const font_instance* font, const rect& bounds, text_alignment align, const std::string str);
    void layout(const rect& bounds);

    void set_text(const std::string str);
    void render();
};

struct button {
    ui_manager* ui;

    rect crad;
    rect bounds;
    vec3 color;

    shader_program shader;
    font_instance* font;
    text content;

    button(ui_manager* ui, const std::string& str, const rect& crad, vec3 color);
    void layout(const rect& bounds);
    bool draw();
};

NAMESPACE_END