#pragma once
#include "utils.hpp"
#include <unordered_map>
#include <string>

NAMESPACE_BEGIN
struct shader_program;

#define version_head "#version 310 es\n"
#define PI_define "#define M_PI 3.1415926535897932384626433832795\n"
#define num_to_str(num) #num

constexpr const char* vert_src = version_head R"(uniform mat4 projection;
        in vec2 position;
        in vec2 uvs;
        out vec2 out_uvs;

        void main() {
             gl_Position = projection * vec4(position, 0, 1);
             out_uvs = uvs;
        }
)";

constexpr const char* vert_quad_src = version_head R"(uniform mat4 projection;
        in vec2 position;
        in vec2 uvs;
        out vec2 out_uvs;

        uniform vec4 transform;

        void main() {
            vec2 real_position = transform.xy + position * transform.zw;
            gl_Position = projection * vec4(real_position, 0, 1);
            out_uvs = uvs;
        }
)";

std::string frag_simple_tex_sampler_src(bool oes_input, u32 binding_slot);

std::string frag_glyph_src(u32 binding_slot);

constexpr const char* vert_instanced_quad_src = version_head R"(uniform mat4 projection;
    layout (location = 0) in vec2 pos;
    layout (location = 1) in vec2 uv;

    layout (location = 2) in vec2 v0;
    layout (location = 3) in vec2 v1;
    layout (location = 4) in vec2 v2;
    layout (location = 5) in vec2 v3;

    layout (location = 6) in vec2 uv_tl;
    layout (location = 7) in vec2 uv_br;

    out vec2 out_uvs;

    void main() {
        vec2 v_pos = mix(mix(v0, v1, pos.x), mix(v3, v2, pos.x), pos.y);
        gl_Position = projection * vec4(v_pos, 0, 1);
        out_uvs = vec2(mix(uv_tl.x, uv_br.x, uv.x), mix(uv_tl.y, uv_br.y, uv.y));
    }
)";

constexpr const char* vert_instanced_point_src = version_head R"(uniform mat4 projection;
    uniform float scale;

    layout (location = 0) in vec2 pos;
    layout (location = 1) in vec2 uv;

    layout (location = 2) in vec2 pos_middle;

    out vec2 out_uvs;

    void main() {
        vec2 v_pos = pos_middle + pos * scale;

        gl_Position = projection * vec4(v_pos, 0, 1);
        out_uvs = uv;
    }
)";

constexpr const char* vert_instanced_line_src = version_head R"(uniform mat4 projection;
    uniform float thickness;

    layout (location = 0) in vec2 pos;
    layout (location = 1) in vec2 uv;

    layout (location = 2) in vec2 pos_start;
    layout (location = 3) in vec2 pos_end;

    out vec2 out_uvs;

    void main() {
        vec2 x_basis = pos_end - pos_start;
        vec2 y_basis = normalize(vec2(-x_basis.y, x_basis.x));

        vec2 v_pos = pos_start + (x_basis * pos.x) + (y_basis * pos.y * thickness);

        gl_Position = projection * vec4(v_pos, 0, 1);
        out_uvs = uv;
    }
)";

std::string frag_gauss_blur_src(bool use_oes, u32 N, const vec2& pixel_shift);

constexpr const char* frag_debug_src = version_head R"(precision mediump float;
        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
             out_col = vec4(out_uvs, 1.0, 1.0);
        }
)";

std::string frag_sampler_src(bool use_oes);

constexpr const char* frag_DEBUG_marker_src = version_head R"(precision mediump float;
        in vec2 out_uvs;
        out vec4 out_col;

        uniform vec3 color;

        void main() {
            float opacity = length(out_uvs - vec2(0.5));
            float alpha = step(1.0 / sqrt(2.0), 1.0 - opacity);
            out_col = vec4(color, alpha);
        }
)";

constexpr const char* frag_border_src = version_head PI_define R"(precision mediump float;
        in vec2 out_uvs;
        out vec4 out_col;

        const float border_fadeout = 0.03;

        uniform float time;

        void main() {
            float inner_core = -0.95;
            float outer_core = +0.95;
            
            float alpha = smoothstep(inner_core - border_fadeout, inner_core, out_uvs.y); 
            alpha -= smoothstep(outer_core, outer_core + border_fadeout, out_uvs.y);

            out_col = vec4(1.0, 1.0, 1.0, 1.0);
        }
)";

constexpr const char* frag_particle_src = version_head R"(precision mediump float;
        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
            float opacity = length(out_uvs - vec2(0.5));
            float alpha = step(1.0 / sqrt(2.0), 1.0 - opacity);
            out_col = vec4(1.0, 1.0, 1.0, alpha);
        }
)";

constexpr const char* frag_shutter_src = version_head R"(precision mediump float;
        in vec2 out_uvs;
        out vec4 out_col;

        uniform float inner_out;

        const float border_fadeout = 0.03;
        const float outer_in = 0.85;
        const float outer_out = 0.95;

        const vec2 min_max_opacity = vec2(0.2, 1.0);

        void main() {
            float len = length(2.0 * out_uvs - vec2(1));
            float alpha = min_max_opacity.y * smoothstep(outer_out + border_fadeout, outer_out, len); 
            alpha -= (min_max_opacity.y - min_max_opacity.x) * smoothstep(outer_in, outer_in - border_fadeout, len);
            alpha += (min_max_opacity.y - min_max_opacity.x) * smoothstep(inner_out + border_fadeout, inner_out, len);

            out_col = vec4(1, 1, 1, alpha);
        }
)";

NAMESPACE_END