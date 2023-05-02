#pragma once
#include "utils.hpp"
#include "backend.hpp"
#include "assets.hpp"
#include <string>
#include <unordered_map>

NAMESPACE_BEGIN

struct ui_manager;

struct glyph {
    s32 i;
    rect uv;
    vec2 off;
    vec2 size;
    f32 x_advance;
};

struct font_instance {
    ui_manager* ui;
    std::unordered_map<s32, glyph> glyph_map;

    svec2 atlas_size;
    u8* atlas;
    texture atlas_texture;
    f32* kerning;

    f32 font_height, line_height;

    font_instance(ui_manager* ui, font_asset_id id, f32 height);

    const glyph* get_glyph(s32 g) const;
    f32 get_kerning(s32 first, s32 next) const;
    void use(s32 slot) const;
};

f32 ease_in_sine(f32 t);

f32 ease_in_out_quad(f32 t);

enum class animation_curve {
    LINEAR = 0, EASE_IN, EASE_IN_OUT
};

enum animation_flags {
    RESET_AFTER_COMPLETION = 1,
    CONTINUE_PLAYING_REVERSED = 2
};

enum animation_state {
    WAITING, STARTED, FINISHED
};

template<typename T>
struct animation {
    engine_backend* backend;

    animation_curve curve;
    T start_value;
    T end_value;
    T value;

    f32 start_time;
    f32 start_delay;
    f32 duration;
    u32 state;
    u32 flags;

    animation(engine_backend* backend, animation_curve curve, T start_value, T end_value, f32 start_delay, f32 duration, u32 flags) 
        : backend(backend), curve(curve), start_value(start_value), end_value(end_value), value(start_value), 
          start_delay(start_delay), duration(duration), state(WAITING), flags(flags) {}

    void start() {
        // todo: fix this if(state != STARTED) backend->running_animations++;
        state = STARTED;

        value = start_value;

        start_time = backend->time;
    }

    void reverse() {
        T temp = start_value;
        start_value = end_value;
        end_value = temp;
    }

    T update() {
        if(state != STARTED) return value;

        f32 time_elapsed = backend->time - start_time;
        f32 t = (time_elapsed - start_delay) / duration;

        if(t < 0.0f) return start_value;
        
        if(t > 1.0f) {
            state = FINISHED;
            // todo: fix thisbackend->running_animations--;

            if(flags & animation_flags::RESET_AFTER_COMPLETION) value = start_value;
            else value = end_value;

            return value;
        }

        if(flags & animation_flags::CONTINUE_PLAYING_REVERSED) {
            t = 1.0f - 2.0f * abs(t - 0.5f);
        }
        
        switch(curve) {
        case animation_curve::LINEAR: break;
        case animation_curve::EASE_IN: {
            t = ease_in_sine(t);
        } break;
        case animation_curve::EASE_IN_OUT: {
            t = ease_in_out_quad(t);
        } break;
        default: LOGE_AND_BREAK("Animation curve is not supported.");
        }

        value = end_value * t + start_value * (1.0f - t); 
        return value;
    }
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

    vec3 line_seperator_color = { 0.3f, 0.3f, 0.3f };
    f32 line_seperator_height = 0.0025f;
    
    ui_theme(bool enable_dark_mode);
};

struct ui_manager {
    engine_backend* backend;
    asset_manager* assets;
    ui_theme theme;
    font_instance* small_font;
    font_instance* middle_font;

    std::unordered_map<u64, font_instance> font_map;

    ui_manager(engine_backend* backend, asset_manager* assets, bool enable_dark_mode);
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

    sdf_animation_asset_id checked_icon;
    animation<f32> check_animation;

    round_checkbox(ui_manager* ui, bool checked);
    void layout(rect bounds);
    void set_checked(bool checked);
    void toggle_checked();

    void draw();
};

struct sdf_image {
    ui_manager* ui;
    rect bounds;
    vec4 color;

    sdf_animation_asset_id id;
    s32 l_depth, c_depth, n_depth;
    animation<f32> blend_animation;

    sdf_image(ui_manager* ui, vec3 color, sdf_animation_asset_id id, f32 blend_duration);
    void layout(rect bounds);
    void next_depth();

    void draw();
};

struct sdf_button {
    sdf_image img;

    sdf_button(ui_manager* ui, vec3 color, sdf_animation_asset_id id);
    void layout(rect bounds);

    bool draw();
};

vec2 map_to_rect(const vec2& pt, const rect* rect);

rect cut_margins(const rect& r, f32 margin);

rect cut_margins(const rect& r, const rect& margin);

rect get_at_top(const rect& r, f32 h);

rect get_at_bottom(const rect& r, f32 h);

rect get_at_left(const rect& r, f32 w);

rect grid_split(const rect& r, s32 i, s32 splits, split_direction dir);

rect get_between(const rect& r, f32 t, f32 b);

rect get_texture_uvs_aligned_top(const rect& r, const svec2& tex_size);

enum class alignment {
    LEFT, RIGHT,
    TOP_RIGHT
};

rect get_texture_aligned_rect(const rect& r, const svec2& size, alignment align);

rect align_rect(const rect& bounds, const rect& r, alignment align);

NAMESPACE_END