#pragma once
#include "types.hpp"
#include "backend.hpp"

#include <vector>

NAMESPACE_BEGIN

extern const s32 points_per_side_incl_start_corner;
extern const s32 points_per_contour;

struct mask_mesher {
    vec2* contour;
    vec2* top_contour;
    vec2* right_contour;
    vec2* bottom_contour;
    vec2* left_contour;
    vertex* vertices;
    svec2 mesh_size;
    svec2 heatmap_size;

    const f32* exists;
    f32* heatmap;

    f32 smoothness;

    void init(const f32* exists, f32* heatmap, const svec2& heatmap_size, f32 smoothness);
    vec2 sample_at(vec2 pt) const;
    void mesh(engine_backend* backend);

    bool does_mesh_exist() const;
};

struct sticky_particle_system {
    const mask_mesher* mesher;

    svec2 size;
    f32 margin;
    f32 particle_size;
    f32 anim_duration;

    vec2* pos_from;
    vec2* pos_to;
    f32 last_pos_reset_time;

    shader_program shader;
    instanced_quads quads;

    void gen_from_and_to();
    void gen_and_fill_quads(const engine_backend* backend);
    void init(engine_backend* backend, const mask_mesher* mesher, svec2 size, f32 margin, f32 particle_size, f32 anim_duration);
    void render(engine_backend* backend);
};

struct mesh_border {
    const vec2* left_border, *top_border, *right_border, *bottom_border;
    s32 border_size;

    std::vector<vertex> mesh_vertices;
    std::vector<u32> mesh_indices;

    shader_program shader;
    shader_buffer buffer;

    variable time_var;

    void gen_and_fill_mesh_vertices();
    void init(engine_backend* backend, const vec2* left_border, const vec2* top_border, const vec2* right_border, const vec2* bottom_border, s32 border_size, shader_buffer buffer);
    void render(f32 time);
};

NAMESPACE_END