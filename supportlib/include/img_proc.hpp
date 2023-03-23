#pragma once
#include "types.hpp"
#include "backend.hpp"

#include <vector>

NAMESPACE_BEGIN

struct mask_mesher {
    vec2* top_contour;
    vec2* right_contour;
    vec2* bottom_contour;
    vec2* left_contour;
    vertex* vertices;
    svec2 mesh_size;

    const f32* exists;
    vec2* contour;
    s32 contour_size;

    void init(const f32* exists, vec2* contour, s32 contour_size);
    void mesh(engine_backend* backend);

    bool does_mesh_exist() const;
};

NAMESPACE_END