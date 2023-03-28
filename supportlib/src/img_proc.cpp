#include "img_proc.hpp"
#include "log.hpp"
#include <algorithm>
#include <math.h>

#include "cam_preview.hpp"

using namespace docscanner;

constexpr f32 binarize_threshold = 0.8f;
constexpr s32 points_per_side = 10;

void mask_mesher::init(const f32* exists, f32* heatmap, s32 contour_size, const svec2& heatmap_size) {
    this->exists = exists;
    this->heatmap = heatmap;
    this->contour_size = contour_size;
    this->mesh_size = { points_per_side, points_per_side };
    this->heatmap_size = heatmap_size;
    
    top_contour = new vec2[points_per_side];
    right_contour = new vec2[points_per_side];
    bottom_contour = new vec2[points_per_side];
    left_contour = new vec2[points_per_side];
    vertices = new vertex[mesh_size.area()];
}

void sample_points_from_contour(vec2* pts, const vec2* contour, s32 contour_size, s32 corner_idx, s32 n) {
    s32 points_per_part = (s32)ceil(n / (f32)points_per_side_incl_start_corner);
    s32 pt_idx = 0;
    s32 points_left = n - 1;

    corner_idx *= points_per_side_incl_start_corner;

    for(s32 i = 0; i < points_per_side_incl_start_corner; i++) {
        s32 c_start = corner_idx + i;
        s32 c_end = c_start + 1;
    
        for(s32 p = 0; p < std::min(points_per_part, points_left); p++) {
            f32 t = p / (f32)(points_per_part - 1);
            vec2 pt = contour[c_end] * t + contour[c_start] * (1.0f - t);
            pts[pt_idx++] = pt;

            points_left--;
        }
    }

    pts[pt_idx++] = contour[(corner_idx + points_per_side_incl_start_corner) % contour_size];
}

vec2* interpolate_lines(const vec2* start, const vec2* end_backwards) {
    vec2* positions = new vec2[points_per_side * points_per_side];

    for(u32 y = 0; y < points_per_side; y++) {
        const vec2& s = start[y];
        const vec2& e = end_backwards[points_per_side - 1 - y];

        for(u32 x = 0; x < points_per_side; x++) {
            positions[x * points_per_side + y] = vec2::lerp(s, e, x / (f32)(points_per_side - 1));
        }
    }

    return positions;
}

void interpolate_mesh(vertex* vertices, const vec2* left, const vec2* top, const vec2* right, const vec2* bottom) {
    auto left_right = interpolate_lines(left, right);
    auto top_bottom = interpolate_lines(top, bottom);

    const s32 n = points_per_side;
    for(s32 y = 0; y < n; y++) {
        for(s32 x = 0; x < n; x++) {
            vertex vert = {
                .pos = left_right[y * n + x] * 0.5f + top_bottom[x * n + (n - 1 - y)] * 0.5f,
                .uv = { x / (f32)(n - 1), y / (f32)(n - 1) }
            };

            vertices[x * n + y] = vert;
        }
    }

    delete left_right;
    delete top_bottom;
}

void mask_mesher::mesh(engine_backend* backend) {
    if (does_mesh_exist()) {
        vec2 contour[contour_size];
        for(s32 c = 0; c < contour_size; c++) {
            f32 max_value = 0.0f;
            s32 max_x = -1, max_y = -1;
        
            for(s32 x = 0; x < heatmap_size.x; x++) {
                for(s32 y = 0; y < heatmap_size.y; y++) {
                    s32 i = x * heatmap_size.y * contour_size + y * contour_size + c;

                    if(heatmap[i] > max_value) {
                        max_value = heatmap[i];
                        max_x = x;
                        max_y = y;
                    }
                }
            }

            contour[c] = { max_x / (f32)(heatmap_size.x - 1), max_y / (f32)(heatmap_size.y - 1) };
        }

        for(s32 i = 0; i < contour_size; i++) {
            vec2& pt = contour[i];
            pt = { 1.0f - pt.x, 1.0f - pt.y };
        }

        sample_points_from_contour(top_contour, contour, contour_size, 0, points_per_side);
        sample_points_from_contour(right_contour, contour, contour_size, 1, points_per_side);
        sample_points_from_contour(bottom_contour, contour, contour_size, 2, points_per_side);
        sample_points_from_contour(left_contour, contour, contour_size, 3, points_per_side);
    
        interpolate_mesh(vertices, left_contour, top_contour, right_contour, bottom_contour);
    }
}

bool mask_mesher::does_mesh_exist() const {
    return true; // (*exists) >= binarize_threshold;
}