#include "img_proc.hpp"
#include "log.hpp"
#include <algorithm>
#include <math.h>
#include <random>

#include "cam_preview.hpp"

using namespace docscanner;

const s32 docscanner::points_per_side_incl_start_corner = 4;
const s32 docscanner::points_per_contour = points_per_side_incl_start_corner * 4;

constexpr f32 binarize_threshold = 0.8f;
constexpr s32 points_per_side = 10;

void mask_mesher::init(const f32* exists, f32* heatmap, const svec2& heatmap_size, f32 smoothness) {
    this->exists = exists;
    this->heatmap = heatmap;
    this->mesh_size = { points_per_side, points_per_side };
    this->heatmap_size = heatmap_size;
    this->smoothness = smoothness;
    
    contour = new vec2[points_per_contour];
    top_contour = new vec2[points_per_side];
    right_contour = new vec2[points_per_side];
    bottom_contour = new vec2[points_per_side];
    left_contour = new vec2[points_per_side];
    vertices = new vertex[mesh_size.area()];
}

void sample_points_from_contour(vec2* pts, const vec2* contour, s32 points_per_contour, s32 corner_idx, s32 n) {
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

    pts[pt_idx++] = contour[(corner_idx + points_per_side_incl_start_corner) % points_per_contour];
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

vec2 mask_mesher::sample_at(vec2 pt) const {
    // ASSERT(pt.x >= 0.0f && pt.y >= 0.0f && pt.x <= 1.0f && pt.y <= 1.0f, "Point is out of bounds.");

    pt.x = clamp(pt.x, 0.0f, 1.0f - 1e-5);
    pt.y = clamp(pt.y, 0.0f, 1.0f - 1e-5);

    f32 x_i, y_i;
    f32 x_t = modff(pt.x * (f32)(mesh_size.x - 1), &x_i);
    f32 y_t = modff(pt.y * (f32)(mesh_size.y - 1), &y_i);

    s32 x = (s32)x_i;
    s32 y = (s32)y_i;

    const vec2& tl = vertices[x * mesh_size.y + y].pos;
    const vec2& tr = vertices[(x + 1) * mesh_size.y + y].pos;
    const vec2& br = vertices[(x + 1) * mesh_size.y + (y + 1)].pos;
    const vec2& bl = vertices[x * mesh_size.y + (y + 1)].pos;

    const vec2 t = tr * x_t + tl * (1.0f - x_t);
    const vec2 b = br * x_t + bl * (1.0f - x_t);

    const vec2 ret = b * y_t + t * (1.0f - y_t);

    return ret;
}
    
void mask_mesher::mesh(engine_backend* backend) {
    if (!does_mesh_exist()) return;

    for(s32 c = 0; c < points_per_contour; c++) {
        f32 max_value = 0.0f;
        s32 max_x = -1, max_y = -1;
        
        for(s32 x = 0; x < heatmap_size.x; x++) {
            for(s32 y = 0; y < heatmap_size.y; y++) {
                s32 i = x * heatmap_size.y * points_per_contour + y * points_per_contour + c;

                if(heatmap[i] > max_value) {
                    max_value = heatmap[i];
                    max_x = x;
                    max_y = y;
                }
            }
        }

        const vec2 pt = { 1.0f - max_x / (f32)(heatmap_size.x - 1), 1.0f - max_y / (f32)(heatmap_size.y - 1) };
        contour[c] = contour[c] * smoothness + pt * (1.0f - smoothness);
    }

    sample_points_from_contour(top_contour, contour, points_per_contour, 0, points_per_side);
    sample_points_from_contour(right_contour, contour, points_per_contour, 1, points_per_side);
    sample_points_from_contour(bottom_contour, contour, points_per_contour, 2, points_per_side);
    sample_points_from_contour(left_contour, contour, points_per_contour, 3, points_per_side);
    
    interpolate_mesh(vertices, left_contour, top_contour, right_contour, bottom_contour);
}

bool mask_mesher::does_mesh_exist() const {
    return true; // (*exists) >= binarize_threshold;
}

void sticky_particle_system::gen_from_and_to() {
    for(s32 x = 0; x < size.x; x++) {
        for(s32 y = 0; y < size.y; y++) {
            f32 x_t0 = random_f32(margin, 1.0f - margin), x_t1 = random_f32(margin, 1.0f - margin);
            f32 y_t0 = random_f32(margin, 1.0f - margin), y_t1 = random_f32(margin, 1.0f - margin);
            LOGI("%f, %f, %f, %f", x_t0, x_t1, y_t0, y_t1);
        
            pos_from[x * size.y + y] = { x_t0, y_t0 };
            pos_to[x * size.y + y] = { x_t1, y_t1 };
        }
    }
}

void sticky_particle_system::gen_and_fill_quads(const engine_backend* backend) {
    f32 t = (backend->time - last_pos_reset_time) / anim_duration;
    t = ease_in_out_quad(t);

    for(s32 x = 0; x < size.x; x++) {
        for(s32 y = 0; y < size.y; y++) {
            const vec2& from = pos_from[x * size.y + y];
            const vec2& to = pos_to[x * size.y + y];

            vec2 curr = vec2::lerp(from, to, t);

            const vec2 uv = {
                (x + curr.x) / (f32)size.x,
                (y + curr.y) / (f32)size.y
            };

            instanced_quad* quad = quads.quads + (x * size.y + y);
            quad->v0 = mesher->sample_at(uv + vec2({-particle_size, -particle_size}));
            quad->v1 = mesher->sample_at(uv + vec2({+particle_size, -particle_size}));
            quad->v2 = mesher->sample_at(uv + vec2({+particle_size, +particle_size}));
            quad->v3 = mesher->sample_at(uv + vec2({-particle_size, +particle_size}));         
        }
    }

    quads.fill();
}

void sticky_particle_system::init(engine_backend* backend, const mask_mesher* mesher, svec2 size, f32 margin, f32 particle_size, f32 anim_duration) {
    this->mesher = mesher;
    this->size = size;
    this->margin = margin;
    this->particle_size = particle_size;
    this->anim_duration = anim_duration;
    this->last_pos_reset_time = -anim_duration;

    shader = backend->compile_and_link(vert_instanced_quad_src, frag_particle_src);
    quads.init(size.area());
    pos_from = new vec2[size.area()];
    pos_to = new vec2[size.area()];
}

void sticky_particle_system::render(engine_backend* backend) {
    if(backend->time > last_pos_reset_time + anim_duration) {
        last_pos_reset_time = backend->time;
        gen_from_and_to();
    }

    gen_and_fill_quads(backend);

    use_program(shader);
    quads.draw();
}

constexpr f32 thickness = 0.1f;
constexpr f32 half_thickness = thickness / 2.0f;

void push_border_vertices(std::vector<vertex>& mesh_vertices, const vec2* border_pts, s32 border_size, s32 i) {
    const vec2& curr = border_pts[i];
    const vec2& last = (i - 1 < 0) ? curr : border_pts[i - 1];
    const vec2& next = (i + 1 > border_size) ? curr : border_pts[i + 1];
            
    const vec2 normal = ((curr - last) * 0.5 + (next - curr) * 0.5).orthogonal().normalize();
    const vec2 half_border = normal * half_thickness;

    const vec2 curr_small = curr - half_border;
    const vec2 curr_large = curr + half_border;

    vec2 ref = { 1.0f, 0.0f };
    f32 curr_angle = vec2::angle_between(curr, ref);

    mesh_vertices.push_back({curr, {0.5f, curr_angle}});
    mesh_vertices.push_back({curr_small, {0.0f, curr_angle}});
    mesh_vertices.push_back({curr_large, {1.0f, curr_angle}});
}

void push_border_vertices_forward(std::vector<vertex>& mesh_vertices, const vec2* border_pts, s32 border_size) {
    for(s32 i = 0; i < border_size; i++) {
        push_border_vertices(mesh_vertices, border_pts, border_size, i);
    }
}

void push_border_vertices_backward(std::vector<vertex>& mesh_vertices, const vec2* border_pts, s32 border_size) {
    for(s32 i = border_size - 1; i >= 0; i--) {
        push_border_vertices(mesh_vertices, border_pts, border_size, i);
    }
}

void mesh_border::gen_and_fill_mesh_vertices() {
    mesh_indices.clear();
    mesh_vertices.clear();

    for(s32 i = 0; i < 4 * border_size - 1; i++) {
        s32 offset_1 = i * 3;
        s32 offset_2 = i * 3 + 3;

        // inner quad
        mesh_indices.push_back(0 + offset_1);
        mesh_indices.push_back(1 + offset_1);
        mesh_indices.push_back(3 + offset_1);

        mesh_indices.push_back(1 + offset_1);
        mesh_indices.push_back(3 + offset_1);
        mesh_indices.push_back(1 + offset_2);
    
        // outer quad
        mesh_indices.push_back(0 + offset_1);
        mesh_indices.push_back(2 + offset_1);
        mesh_indices.push_back(0 + offset_2);

        mesh_indices.push_back(2 + offset_1);
        mesh_indices.push_back(0 + offset_2);
        mesh_indices.push_back(2 + offset_2);
    }

    // to make it seem complete

    push_border_vertices_forward(mesh_vertices, left_border, border_size);
    push_border_vertices_forward(mesh_vertices, top_border, border_size);
    push_border_vertices_forward(mesh_vertices, right_border, border_size);
    push_border_vertices_forward(mesh_vertices, bottom_border, border_size);
    
    #if false
    push_border_vertices_backward(mesh_vertices, border_vertices, 
        (border_size.y - 1), 
        border_size.y, 
        (border_size.y - 1) + (border_size.x - 1) * border_size.y
    );

    push_border_vertices_backward(mesh_vertices, border_vertices, 
        0, 
        1, 
        (border_size.y - 1)
    );
    #endif

    fill_shader_buffer(buffer, mesh_vertices.data(), mesh_vertices.size() * sizeof(vertex), mesh_indices.data(), mesh_indices.size() * sizeof(u32));
}

void docscanner::mesh_border::init(engine_backend* backend, const vec2* left_border, const vec2* top_border, const vec2* right_border, const vec2* bottom_border, s32 border_size, shader_buffer buffer) {
    this->left_border = left_border;
    this->top_border = top_border;
    this->right_border = right_border;
    this->bottom_border = bottom_border;
    this->border_size = border_size;    
    this->buffer = buffer;

    shader = backend->compile_and_link(vert_src, frag_border_src);

    time_var = get_variable(shader, "time");
}

void mesh_border::render(f32 time) {
    use_program(shader);

    gen_and_fill_mesh_vertices();
    bind_shader_buffer(buffer);

    time_var.set_f32(time);

    glDrawElements(GL_TRIANGLES, mesh_indices.size(), GL_UNSIGNED_INT, null);
}