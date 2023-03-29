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
    void mesh(engine_backend* backend);

    bool does_mesh_exist() const;
};

NAMESPACE_END