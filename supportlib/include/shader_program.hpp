#pragma once
#include "types.hpp"
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

std::string frag_simple_tex_sampler_src(bool oes_input, u32 binding_slot);

constexpr const char* vert_instanced_quad_src = version_head R"(uniform mat4 projection;

    struct quad {
        vec4 tl, tr, br, bl;
    };

    uniform vec4 transforms[%u][4];

    layout (location = 0) in vec2 pos;
    layout (location = 0) in vec2 uv;

    out vec2 out_uv;

    void main() {
        vec4 pts[4] = transforms[gl_InstanceID];
        gl_Position = projection * vec4(pos, 0, 1);
        out_uv = uv;
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

constexpr const char* frag_border_src = version_head PI_define R"(precision mediump float;
        in vec2 out_uvs;
        out vec4 out_col;

        const float border_fadeout = 0.03;
        const float border_pos = 0.5;
        const vec2 border_core_min_max = vec2(0.05, 0.1);
        const float border_interval = 0.5;

        uniform float time;

        void main() {
            float offset = 0.5 * sin(M_PI * time / border_interval) + 1.0;
            float border_core = mix(border_core_min_max.x, border_core_min_max.y, offset);
            
            float inner_core = border_pos - border_core / 2.0;
            float outer_core = border_pos + border_core / 2.0;
            
            float alpha = smoothstep(inner_core - border_fadeout, inner_core, out_uvs.x); 
            alpha -= smoothstep(outer_core, outer_core + border_fadeout, out_uvs.x);

            out_col = vec4(1.0, 1.0, 1.0, alpha);
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

struct shader_programmer {
    std::unordered_map<u64, u32> shader_map;
    std::unordered_map<u64, u32> program_map;

    shader_program compile_and_link(const std::string& vert_src, const std::string& frag_src);
    shader_program compile_and_link(const std::string& comp_src);
};

NAMESPACE_END