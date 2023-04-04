#pragma once
#include "utils.hpp"
#include "backend.hpp"

#include <vector>

NAMESPACE_BEGIN

#ifdef ANDROID
#define CAM_USES_OES_TEXTURE true
#elif defined(LINUX)
#define CAM_USES_OES_TEXTURE false
#else
#error "Platform not supported yet."
#endif

extern const s32 points_per_side_incl_start_corner;
extern const s32 points_per_contour;

struct mask_mesher {
    rect point_range, point_dst;

    vec2* contour;
    vec2* top_contour;
    vec2* right_contour;
    vec2* bottom_contour;
    vec2* left_contour;

    const f32* exists;
    f32* heatmap;
    svec2 heatmap_size;

    svec2 mesh_size;
    vertex* vertices;
    vertex* blend_vertices;
    vec2* blend_to_vertices;

    std::vector<u32> mesh_indices;

    f32 smoothness;

    void init(const f32* exists, f32* heatmap, const svec2& heatmap_size, const rect& point_range, const rect& point_dst, f32 smoothness);
    vec2 sample_at(vec2 pt) const;
    void mesh(engine_backend* backend);
    void blend(f32 t);
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
    const mask_mesher* mesher;

    vec2* points;
    s32 points_size;
    svec2 size;

    lines border_lines;

    variable time_var;
    variable thickness_var;

    void gen_and_fill_lines();
    void init(engine_backend* backend, const mask_mesher* mesher, svec2 size, f32 thickness);
    void render(f32 time);
};

struct mesh_cutout {
    engine_backend* backend;
    const mask_mesher* mesher;

    shader_buffer buffer;
    shader_program shader;

    void gen_and_fill_mesh();
    void init(engine_backend* backend, const mask_mesher* mesher);
    void render(f32 time);
};

NAMESPACE_END