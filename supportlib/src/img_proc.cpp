#include "img_proc.hpp"
#include "log.hpp"
#include <algorithm>
#include <math.h>

using namespace docscanner;

constexpr const char vert_src[] = R"(#version 310 es
        uniform mat4 projection;

        in vec2 position;
        in vec2 uvs;
        out vec2 out_uvs;

        void main() {
             gl_Position = projection * vec4(position, 0, 1);
             out_uvs = uvs;
        }
)";

constexpr const char frag_debug_src[] = R"(#version 310 es
        precision mediump float;

        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
             out_col = vec4(out_uvs, 1.0, 1.0);
        }
)";

constexpr f32 binarize_threshold = 0.8f;
constexpr s32 points_per_side = 10;

void mask_mesher::init(f32* mask_buffer, const svec2& mask_size, float* projection) {
    this->mask_buffer = mask_buffer;
    this->mask_size = mask_size;

    mesh_buffer = make_shader_buffer();
    mesh_program = compile_and_link_program(vert_src, frag_debug_src, null, null);

    use_program(mesh_program);

    get_variable(mesh_program, "projection").set_mat4(projection);

    for(s32 y = 0; y < points_per_side - 1; y++) {
        for(s32 x = 0; x < points_per_side - 1; x++) {
#define TO_INDEX(pt) pt.x * points_per_side + pt.y
            mesh_indices.push_back(TO_INDEX(svec2({x, y})));
            mesh_indices.push_back(TO_INDEX(svec2({x, y + 1})));
            mesh_indices.push_back(TO_INDEX(svec2({x + 1, y})));

            mesh_indices.push_back(TO_INDEX(svec2({x + 1, y})));
            mesh_indices.push_back(TO_INDEX(svec2({x, y + 1})));
            mesh_indices.push_back(TO_INDEX(svec2({x + 1, y + 1})));
#undef TO_INDEX
        }
    }
}

bool is_in_list(std::vector<svec2> list, svec2 pt) {
    for(svec2 lpt: list) {
        if(lpt.x == pt.x && lpt.y == pt.y) return true;
    }

    return false;
}

bool is_surrounded(const f32* buff, const svec2& size, svec2 pt) {
#define get_at(pt) buff[pt.x * size.y + pt.y] 
    f32 sum = 0.0;
    sum += get_at(svec2({pt.x - 1, pt.y}));
    sum += get_at(svec2({pt.x + 1, pt.y}));
    sum += get_at(svec2({pt.x, pt.y - 1}));
    sum += get_at(svec2({pt.x, pt.y + 1}));
    sum += get_at(svec2({pt.x - 1, pt.y + 1}));
    sum += get_at(svec2({pt.x + 1, pt.y + 1}));
    sum += get_at(svec2({pt.x - 1, pt.y - 1}));
    sum += get_at(svec2({pt.x + 1, pt.y - 1}));
#undef get_at

    return sum == 8.0;
}

std::vector<svec2> find_contours(const f32* buff, const svec2& size) {
    svec2 start_pt = { -1, -1 };
    // figure out a starting point on the contour
    for(s32 x = 0; x < size.x; x++) {
        for(s32 y = 0; y < size.y; y++) {
            s32 i = x * size.y + y;
            if (buff[i] == 1.0f) {
                start_pt = { x, y };
                break;
            }
        }
    }

    // return empty contour
    std::vector<svec2> pts;
    if (start_pt.x == -1) return pts;

    // follow contour now
    svec2 pt = start_pt;

    do {
        pts.push_back(pt); \
    
#define check_and_follow(index, new_pt) \
        s32 i_##index = new_pt.x * size.y + new_pt.y; \
        if (buff[i_##index] == 1.0f && !is_in_list(pts, new_pt) && !is_surrounded(buff, size, new_pt)) { \
            pt = new_pt; \
            continue; \
        }    

        check_and_follow(0, svec2({pt.x, pt.y + 1}));
        check_and_follow(1, svec2({pt.x + 1, pt.y}));
        check_and_follow(2, svec2({pt.x, pt.y - 1}));
        check_and_follow(3, svec2({pt.x - 1, pt.y}));

#undef check_and_follow

        break;
    } while (pt.x != start_pt.x || pt.y != start_pt.y);

    return pts;
}


// An implementation of the Ramer–Douglas–Peucker algorithm.
f32 perpendicular_distance(svec2 start, svec2 end, svec2 p) {
    f32 a = (end.x - start.x) * (start.y - p.y) - (start.x - p.x) * (end.y - start.y);
    f32 b = sqrt((end.x - start.x) * (end.x - start.x) + (end.y - start.y) * (end.y - start.y));
    return std::abs(a) / b;
}

f32 arc_length(std::vector<svec2> boundary) {
    f32 len = 0.0f;

    for(u32 i = 0; i < boundary.size(); i++) {
        svec2 s = boundary[i];
        svec2 e = boundary[(i + 1) % boundary.size()];
        svec2 d = { e.x - s.x, e.y - s.y };
        len += sqrt(d.x * d.x + d.y * d.y);
    }

    return len;
}

std::vector<s32> contour_approx(const std::vector<svec2>& boundary, s32 s, s32 e, f32 epsilon) {
    if (s == e) {
        return {s};
    }

    // find point with max distance
    f32 max_dist = 0.0;
    s32 max_i = s;
    for(s32 i = s + 1; i < e; i++) {
        f32 dist = perpendicular_distance(boundary[s], boundary[e], boundary[i]);
        
        if(dist > max_dist) {
            max_dist = dist;
            max_i = i;
        }
    }

    if (max_dist < epsilon) {
        return { s, e };
    }

    auto left = contour_approx(boundary, s, max_i, epsilon);
    auto right = contour_approx(boundary, max_i, e, epsilon);

    std::vector<s32> out;
    for (s32 i : left) out.push_back(i);
    for (s32 i: right) out.push_back(i);
    return out;
}

f32 edge_error(s32 s, s32 e, const std::vector<svec2>& boundary) {
    f32 sum = 0.0f;

    if (e < s) e += boundary.size();

    for(s32 i = s + 1; i < e; i++) {
        sum += perpendicular_distance(boundary[s], boundary[e], boundary[i % boundary.size()]);
    }

    return sum;
}

std::vector<s32> find_most_promising_corner_pts(const std::vector<svec2>& boundary, const std::vector<s32>& candidates) {
    s32 c0 = -1, c1 = -1, c2 = -1, c3 = -1;
    f32 min_error = 999999.0f;

    for(s32 a = 0; a < candidates.size(); a++) {
        for(s32 b = a + 1; b < candidates.size(); b++) {
            for(s32 c = b + 1; c < candidates.size(); c++) {
                for(s32 d = c + 1; d < candidates.size(); d++) {
                    s32 i0 = candidates[a];
                    s32 i1 = candidates[b];
                    s32 i2 = candidates[c];
                    s32 i3 = candidates[d];

                    f32 error = 0.0f;
                    error += edge_error(i0, i1, boundary);
                    error += edge_error(i1, i2, boundary);
                    error += edge_error(i2, i3, boundary);
                    error += edge_error(i3, i0, boundary);
                
                    if (error < min_error) {
                        min_error = error;
                        c0 = i0;
                        c1 = i1;
                        c2 = i2;
                        c3 = i3;
                    }
                }
            }
        }
    
    }

    return { c0, c1, c2, c3 };
}


s32 most_likely_corner(svec2 s0, svec2 s1) {
#define loss(v, d) abs(abs(svec2::dot(v, d) / v.length()) - 1.0f)
#define left_loss(v) loss(v, svec2({ -1, 0 }))
#define top_loss(v) loss(v, svec2({ 0, 1 }))
#define right_loss(v) loss(v, svec2({ 1, 0 }))
#define bottom_loss(v) loss(v, svec2({ 0, -1 }))

    f32 dots[] = {
        left_loss(s0),
        top_loss(s0),
        right_loss(s0),
        bottom_loss(s0)
    };

    s32 s0_i; 
    f32 max_val = 0.0;
    for (s32 i = 0; i < 4; i++) {
        if(dots[i] > max_val) {
            s0_i = i;
            max_val = dots[i];
        }
    }

    s32 lr_i, tb_i;

    if (s0_i == 0 || s0_i == 2) {
        lr_i = (s0_i == 0);
        tb_i = (top_loss(s1) < bottom_loss(s1));
    } else {
        tb_i = (s0_i == 1);
        lr_i = (left_loss(s1) < right_loss(s1)); 
    }

    if (lr_i == 0) {
        if (tb_i == 0) return 0;
        else return 3;
    } else {
        if (tb_i == 0) return 1;
        else return 2;
    }
}

std::vector<s32> reoder_corner_pts_to_match_direction(const std::vector<svec2>& boundary, const std::vector<s32>& corner_pts) {
    std::vector<s32> already_chosen;

    const svec2& c0 = boundary[corner_pts[0]];
    const svec2& c1 = boundary[corner_pts[1]];
    const svec2& c2 = boundary[corner_pts[2]];
    
    s32 real_c0 = most_likely_corner((c1 - c0).orthogonal(), (c2 - c1).orthogonal());

    std::vector<s32> reordered;
    reordered.reserve(4);

    for (s32 i = 0; i < 4; i++) {
        reordered[(i + real_c0) % 4] = corner_pts[i];
    }
    
    return reordered;
}

std::vector<svec2> sample_points_from_boundary(const std::vector<svec2>& boundary, u32 from, u32 to, u32 n) {
    std::vector<svec2> pts;

    if (to < from) to += boundary.size();

    for(u32 i = 0; i < n; i++) {
        u32 b = from + (u32)((i / (f32)(n - 1)) * (to - from));
        pts.push_back(boundary[b % boundary.size()]);
    }

    return pts;
}

vec2* interpolate_lines(const std::vector<svec2>& start, const std::vector<svec2>& end, s32 n) {
    ASSERT(start.size() == end.size(), "Two lines of different lengths can't be interpolated.");

    vec2* positions = new vec2[start.size() * n];

    for(u32 y = 0; y < start.size(); y++) {
        const svec2& s = start[y];
        const svec2& e = end[y];

        for(u32 x = 0; x < n; x++) {
            positions[x * n + y] = svec2::lerp(s, e, x / (f32)(n - 1));
        }
    }

    return positions;
}

void interpolate_mesh(vertex* vertices, const std::vector<svec2>& left, const std::vector<svec2>& top, const std::vector<svec2>& right, const std::vector<svec2>& bottom) {
    s32 n = points_per_side;
    auto left_right = interpolate_lines(left, right, n);
    auto top_bottom = interpolate_lines(top, bottom, n);

    for(s32 y = 0; y < n; y++) {
        for(s32 x = 0; x < n; x++) {
            vertex vert = {
                .pos = (left_right[y * n + x] * 0.5f + top_bottom[x * n + (n - 1 - y)] * 0.5f) * (1.0f / 64.0f),
                .uv = { x / (f32)(n - 1), y / (f32)(n - 1) }
            };

            vertices[x * n + y] = vert;
        }
    }

    delete left_right;
    delete top_bottom;
}

void mask_mesher::mesh() {
    for(s32 i = 0; i < mask_size.x * mask_size.y; i++) {
        mask_buffer[i] = (mask_buffer[i] > binarize_threshold) ? 1.0f : 0.0f;
    }

    auto contour = find_contours(mask_buffer, mask_size);

    if (contour.size() > 0) {
        f32 arc_len = arc_length(contour);
        auto approx_contour = contour_approx(contour, 0, contour.size() - 1, 0.05f * arc_len);

        if(approx_contour.size() >= 4) {
            auto corner_pts = find_most_promising_corner_pts(contour, approx_contour);
            corner_pts = reoder_corner_pts_to_match_direction(contour, corner_pts);

            auto left = sample_points_from_boundary(contour, corner_pts[0], corner_pts[1], points_per_side);
            auto top = sample_points_from_boundary(contour, corner_pts[1], corner_pts[2], points_per_side);
            auto right = sample_points_from_boundary(contour, corner_pts[2], corner_pts[3], points_per_side);
            auto bottom = sample_points_from_boundary(contour, corner_pts[3], corner_pts[0], points_per_side);

            std::reverse(right.begin(), right.end());
            std::reverse(bottom.begin(), bottom.end());

            mesh_vertices.clear();
            for(u32 i = 0; i < points_per_side * points_per_side; i++) 
                mesh_vertices.push_back({});
    
            interpolate_mesh(mesh_vertices.data(), left, top, right, bottom);

            s32 n = points_per_side;

            for(s32 y = 0; y < n; y++) {
                for(s32 x = 0; x < n; x++) {
                    vec2& pos = mesh_vertices[x * n + y].pos;
                    pos = { 1.0f - pos.x, 1.0f - pos.y };
                }
            }

            fill_shader_buffer(mesh_buffer, mesh_vertices.data(), mesh_vertices.size() * sizeof(vertex), mesh_indices.data(), mesh_indices.size() * sizeof(u32));
        }
    }
}