#include "cam_preview.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "android_camera.hpp"

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

constexpr const char frag_src[] = R"(#version 310 es
        precision mediump float;

        uniform layout(binding = 0) sampler2D mask_sampler;
        uniform layout(binding = 1) sampler2D cam_sampler;

        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
             out_col = texture(mask_sampler, out_uvs); // texture(cam_sampler, out_uvs) * texture(mask_sampler, out_uvs).r;
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

void docscanner::cam_preview::pre_init(uvec2 preview_size, int* cam_width, int* cam_height) {
    this->preview_size = preview_size;

    cam = find_and_open_back_camera(preview_size, cam_tex_size);
    *cam_width = (int) cam_tex_size.x;
    *cam_height = (int) cam_tex_size.y;

    LOGI("cam_tex_size: (%u, %u)", cam_tex_size.x, cam_tex_size.y);
}

void docscanner::cam_preview::init_backend(file_context* file_ctx) {
    LOGI("preview_size: (%u, %u)", preview_size.x, preview_size.y);

    preview_program = compile_and_link_program(vert_src, frag_src, nullptr, nullptr);
    ASSERT(preview_program.program, "Preview program could not be compiled.");

    // buffer stuff
    float p = (cam_tex_size.x / (float) cam_tex_size.y) * (preview_size.x / (float) preview_size.y);
    cam_tex_left = (1.0f - p) / 2.0f;
    cam_tex_right = 1.0f - cam_tex_left;
    
    vertex vertices[] = {
        {{1.f, 0.f}, {1, 0}},
        {{0.f, 1.f}, {0, 1}},
        {{1.f, 1.f}, {0, 0}},
        {{0.f, 0.f}, {1, 1}}
    };

    u32 indices[] = { 
        0, 1, 2, 
        0, 3, 1 
    };

    use_program(preview_program);

    auto proj_matrix_var = get_variable(preview_program, "projection");
    
    float projection[16];
    mat4f_load_ortho(cam_tex_left, cam_tex_right, 0.0f, 1.0f, -1.0f, 1.0f, projection);

    proj_matrix_var.set_mat4(projection);

    cam_quad_buffer = make_shader_buffer();
    fill_shader_buffer(cam_quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    uvec2 downsampled_size = {64, 64};

    nn_input_buffer_size = downsampled_size.x * downsampled_size.y * 4 * sizeof(float);
    nn_input_buffer = new u8[nn_input_buffer_size];

    nn_output_buffer_size = downsampled_size.x * downsampled_size.y * 1 * sizeof(float);
    nn_output_buffer = new u8[nn_output_buffer_size];
    
    nn_output_tex = create_texture(downsampled_size , GL_R32F);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    tex_downsampler.init(cam_tex_size, downsampled_size, true, null, 2.0);

    nn = create_neural_network_from_path(file_ctx, "seg_model.tflite", execution_pref::sustained_speed);

    test_rect_buffer = make_shader_buffer();

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

    is_init = true;
}

void docscanner::cam_preview::init_cam(ANativeWindow* texture_window) {
    init_camera_capture_to_native_window(cam, texture_window);
}

#include <chrono>
#include <vector>

bool is_in_list(std::vector<svec2> list, svec2 pt) {
    for(svec2 lpt: list) {
        if(lpt.x == pt.x && lpt.y == pt.y) return true;
    }

    return false;
}

bool is_surrounded(const f32* buff, const uvec2& size, svec2 pt) {
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

std::vector<svec2> find_contours(const f32* buff, const uvec2& size) {
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

#include <algorithm>
#include <math.h>

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

std::vector<s32> contour_approx(std::vector<svec2> boundary, s32 s, s32 e, f32 epsilon) {
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

#if false
std::vector<s32> find_most_promising_corner_pts(std::vector<svec2> boundary, std::vector<s32> candidates) {
    std::vector<std::tuple<f32, s32>> to_be_sorted;

    for(s32 i: candidates) {
        s32 l = (i - 1 < 0) ? boundary.size() - 1 : (i - 1);
        s32 n = (i + 1) % boundary.size();
        
        svec2 lmi = boundary[l] - boundary[i];
        svec2 nmi = boundary[n] - boundary[i];

        f32 angle = svec2::angle_between(lmi, nmi);
        angle = abs(angle - M_PI_2);

        to_be_sorted.push_back(std::make_tuple(angle, i));
    }

    std::sort(to_be_sorted.begin(), to_be_sorted.end());

    return { std::get<1>(to_be_sorted[0]), std::get<1>(to_be_sorted[1]), std::get<1>(to_be_sorted[2]), std::get<1>(to_be_sorted[3]) };
}
#else
f32 edge_error(s32 s, s32 e, std::vector<svec2> boundary) {
    f32 sum = 0.0f;

    if (e < s) e += boundary.size();

    for(s32 i = s + 1; i < e; i++) {
        sum += perpendicular_distance(boundary[s], boundary[e], boundary[i % boundary.size()]);
    }

    return sum;
}

std::vector<s32> find_most_promising_corner_pts(std::vector<svec2> boundary, std::vector<s32> candidates) {
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
#endif

std::vector<svec2> sample_points_from_boundary(std::vector<svec2> boundary, u32 from, u32 to, u32 n) {
    std::vector<svec2> pts;

    if (to < from) to += boundary.size();

    for(u32 i = 0; i < n; i++) {
        u32 b = from + (u32)((i / (f32)(n - 1)) * (to - from));
        pts.push_back(boundary[b % boundary.size()]);
    }

    return pts;
}

vec2* interpolate_lines(std::vector<svec2> start, std::vector<svec2> end, s32 n) {
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

void interpolate_mesh(vertex* vertices, std::vector<svec2> left, std::vector<svec2> top, std::vector<svec2> right, std::vector<svec2> bottom) {
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

#include <string.h>

void docscanner::cam_preview::render() {
    if(!is_init) return;

    nn_input_tex = tex_downsampler.downsample();

    get_framebuffer_data(tex_downsampler.output_fb, tex_downsampler.output_size, nn_input_buffer, nn_input_buffer_size);
    
    invoke_neural_network_on_data(nn, nn_input_buffer, nn_input_buffer_size, nn_output_buffer, nn_output_buffer_size);

    auto* buff = reinterpret_cast<f32*>(nn_output_buffer);
    for(u32 i = 0; i < tex_downsampler.output_size.x * tex_downsampler.output_size.y; i++) {
        buff[i] = (buff[i] > binarize_threshold) ? 1.0f : 0.0f;
    }

    auto contour = find_contours(buff, tex_downsampler.output_size);

    if (contour.size() > 0) {
        f32 arc_len = arc_length(contour);
        auto approx_contour = contour_approx(contour, 0, contour.size() - 1, 0.05f * arc_len);

        if(approx_contour.size() >= 4) {
            auto corner_pts = find_most_promising_corner_pts(contour, approx_contour);

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
                    auto v = mesh_vertices[x * n + y].pos * tex_downsampler.output_size.y;
                    buff[(s32)v.x * tex_downsampler.output_size.y + (s32)v.y] = mesh_vertices[x * n + y].pos.x;
                    mesh_vertices[x * n + y].pos = { 1.0f - mesh_vertices[x * n + y].pos.x, 1.0f - mesh_vertices[x * n + y].pos.y };
                }
            }

            fill_shader_buffer(mesh_buffer, mesh_vertices.data(), mesh_vertices.size() * sizeof(vertex), mesh_indices.data(), mesh_indices.size() * sizeof(u32));
        }
    }

    set_texture_data(nn_output_tex, nn_output_buffer, tex_downsampler.output_size.x, tex_downsampler.output_size.y);

    canvas c = {
        .bg_color={0, 1, 0}
    };
    
    use_program(preview_program);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(cam_quad_buffer.id);
    
    bind_texture_to_slot(0, nn_output_tex);
    bind_texture_to_slot(1, *nn_input_tex);
    draw(c);


    use_program(mesh_program);

    glBindVertexArray(mesh_buffer.id);
    glDrawElements(GL_TRIANGLES, mesh_indices.size(), GL_UNSIGNED_INT, null);


    /*glBindVertexArray(test_rect_buffer.id);
    
    vertex vertices[] = {
        {tr, {1, cam_tex_left}},   // 1, 0
        {bl, {0, cam_tex_right}},  // 0, 1
        {br, {0, cam_tex_left}},   // 1, 1
        {tl, {1, cam_tex_right}}  // 0, 0
    };

    u32 indices[] = { 
        0, 1, 2, 
        0, 3, 1 
    };

    fill_shader_buffer(cam_quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    draw(c);*/

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    auto dur = end - last_time;
    LOGI("frame time: %lldms", dur);
    last_time = end;
}