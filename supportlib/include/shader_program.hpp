#pragma once
#include "types.hpp"
#include <unordered_map>
#include <string>

NAMESPACE_BEGIN
struct shader_program;

#define version_head "#version 310 es\n"
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

struct shader_programmer {
    std::unordered_map<u64, u32> shader_map;
    std::unordered_map<u64, u32> program_map;

    shader_program compile_and_link(const std::string& vert_src, const std::string& frag_src);
    shader_program compile_and_link(const std::string& comp_src);
};

NAMESPACE_END