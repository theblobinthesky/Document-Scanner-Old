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

struct ui_manager {
    engine_backend* backend;

    std::unordered_map<u64, font_instance> font_map;

    ui_manager(engine_backend* backend);
    font_instance* get_font(const std::string& path, f32 size);
};

struct text {
    engine_backend* backend;

    vec2 pos;
    const font_instance* font;
    std::string str;

    shader_program shader;
    instanced_quads quads;

    void init(engine_backend* backend, const font_instance* font, const vec2& pos, const std::string str);

    void set_text(const std::string str);
    void render();
};

NAMESPACE_END