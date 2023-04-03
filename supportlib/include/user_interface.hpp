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

    void init(engine_backend* backend, const std::string& path, f32 height);

    const glyph* get_glyph(s32 g) const;
    f32 get_kerning(s32 first, s32 next) const;
    void use(s32 slot) const;
};

struct ui_theme {
    vec3 black = { 0, 0, 0 };
    vec3 background_color;
    vec3 primary_color;
    vec3 primary_dark_color;
    
    void init(bool enable_dark_mode);
};

struct ui_manager {
    engine_backend* backend;
    ui_theme theme;

    std::unordered_map<u64, font_instance> font_map;

    ui_manager(engine_backend* backend);
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

    void init(engine_backend* backend, const font_instance* font, const rect& bounds, text_alignment align, const std::string str);

    void set_text(const std::string str);
    void render();
};

struct button {
    ui_manager* ui;

    rect bounds;

    shader_program shader;
    font_instance* font;
    text content;

    void init(ui_manager* ui, const rect& bounds, const std::string& str);
    void draw();
};

NAMESPACE_END