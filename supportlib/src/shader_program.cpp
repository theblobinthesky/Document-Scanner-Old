#include "shader_program.hpp"
#include "backend.hpp"
#include <functional>
#include <math.h>

using namespace docscanner;

#define version_head "#version 310 es\n"
#define precision_head "precision mediump float;\n"
#define version_and_precision_head version_head precision_head
#define define_PI "#define M_PI 3.1415926535897932384626433832795\n"
#define define_edge_softness "const float edge_softness = 0.002;\n"
#define num_to_str(num) #num

std::string docscanner::vert_src() {
    return version_and_precision_head R"(uniform mat4 projection;
        in vec2 position;
        in vec2 uvs;
        out vec2 out_uvs;

        void main() {
             gl_Position = projection * vec4(position, 0, 1);
             out_uvs = uvs;
        }
    )";
}

std::string docscanner::vert_quad_src() {
    return version_and_precision_head R"(uniform mat4 projection;
        in vec2 position;
        in vec2 uvs;
        out vec2 out_rel_pos;
        out vec2 out_uvs;

        uniform vec4 bounds;
        uniform vec4 uv_bounds;
        uniform bool rot_uv_90_deg;

        void main() {
            vec2 real_position = mix(bounds.xy, bounds.zw, position);
            vec2 real_uvs = mix(uv_bounds.xy, uv_bounds.zw, rot_uv_90_deg ? vec2(uvs.y, uvs.x) : uvs);
            
            gl_Position = projection * vec4(real_position, 0, 1);
            out_rel_pos = uvs;
            out_uvs = real_uvs;
        }
    )";
}

std::string docscanner::vert_instanced_quad_src() {
    return version_and_precision_head R"(uniform mat4 projection;
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
}

std::string docscanner::vert_instanced_point_src() {
    return version_and_precision_head R"(uniform mat4 projection;
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
}

std::string docscanner::vert_instanced_line_src() {
    return version_and_precision_head R"(uniform mat4 projection;
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
}

std::string get_sampler_src(bool use_oes, u32 binding_slot) {
    return use_oes ? 
    R"( 
        #extension GL_OES_EGL_image_external_essl3 : require
        uniform layout(binding = )" + std::to_string(binding_slot) + R"() samplerExternalOES sampler;
    )" : 
    R"(
        uniform layout(binding = )" + std::to_string(binding_slot) + R"() sampler2D sampler;
    )";
}

std::string get_array_sampler_src(u32 binding_slot) {
    return 
    R"(
        uniform layout(binding = )" + std::to_string(binding_slot) + R"() sampler2DArray sampler;
    )";
}

std::string docscanner::frag_glyph_src(u32 binding_slot) {
    return version_head + get_sampler_src(false, binding_slot) + precision_head + R"(
            in vec2 out_uvs;
            out vec4 out_col;

            uniform vec4 color;

            void main() {
                float glyph_alpha = texture(sampler, out_uvs).r;
                out_col = color * glyph_alpha;
            }
        )";
}

std::string compute_gauss_coefficients(s32 N, s32 M) {
    f32 sigma = (N - 1.0f) / 4.0f;
    
    f32 *coeffs = new f32[M];
    f32 sum = 0.0f;

#define compute_gauss_coeff(i) \
        const f32 gauss_factor = 1.0 / sqrt(2 * M_PI * sigma); \
        const f32 x = (f32)(i); \
        coeffs[i] = gauss_factor * pow(M_E, -(x * x) / (2 * sigma * sigma));

    compute_gauss_coeff(0);    
    sum += coeffs[0];    

    for(int i = 1; i < M; i++) {
        compute_gauss_coeff(i);
        sum += 2 * coeffs[i];
    } 

#undef compute_gauss_coeff

    // normalize
    for(int i = 0; i < M; i++) {
        coeffs[i] /= sum;
    }

    std::string ret = "";
    for(int i = 0; i < M; i++) {
        ret = ret + std::to_string(coeffs[i]);

        if(i != M - 1) ret = ret + ", ";
    }

    delete[] coeffs;

    return ret;
}

std::string docscanner::frag_gauss_blur_src(bool use_oes, u32 N, const vec2& pixel_shift) {
    u32 M = N / 2 + 1;

    return version_head + get_sampler_src(use_oes, 0) + precision_head + R"(
        in vec2 out_uvs;
        out vec4 out_col;

        const int M = )" + std::to_string(M) + R"(;
        const float coeffs[M] = float[M]()" + compute_gauss_coefficients(N, M) + R"();
        const vec2 pixel_shift = vec2()" + std::to_string(pixel_shift.x) + ", " + std::to_string(pixel_shift.y) +  R"();

        void main() {
            vec4 col = coeffs[0] * texture(sampler, out_uvs);

            for(int i = 1; i < M; i += 2) {
                float w0 = coeffs[i];
                float w1 = coeffs[i + 1];

                float w = w0 + w1;
                float t = w1 / w;

                col += w * texture(sampler, out_uvs + (float(i) + t) * pixel_shift);
                col += w * texture(sampler, out_uvs - (float(i) + t) * pixel_shift);
            }

            out_col = col;
        }
    )";
}

std::string docscanner::frag_debug_src() {
    return version_and_precision_head R"(
            in vec2 out_uvs;
            out vec4 out_col;

            void main() {
                 out_col = vec4(out_uvs, 1.0, 1.0);
            }
    )";
}


std::string docscanner::frag_sampler_src(bool use_oes) {
    return version_head + get_sampler_src(use_oes, 0) + precision_head + R"(
        in vec2 out_uvs;
        out vec4 out_col;
    
        void main() {
             out_col = texture(sampler, vec2(out_uvs.y, 1.0 - out_uvs.x));
        }
    )";
}

std::string docscanner::frag_DEBUG_marker_src() {
    return version_and_precision_head R"(
        in vec2 out_uvs;
        out vec4 out_col;

        uniform vec3 color;

        void main() {
            float opacity = length(out_uvs - vec2(0.5));
            float alpha = step(1.0 / sqrt(2.0), 1.0 - opacity);
            out_col = vec4(color, alpha);
        }
    )";
}

std::string docscanner::frag_border_src() {
    return version_and_precision_head define_PI R"(
        in vec2 out_uvs;
        out vec4 out_col;

        const float border_fadeout = 0.03;

        uniform float time;
        uniform vec4 color;

        void main() {
            float inner_core = -0.95;
            float outer_core = +0.95;
            
            float alpha = smoothstep(inner_core - border_fadeout, inner_core, out_uvs.y); 
            alpha -= smoothstep(outer_core, outer_core + border_fadeout, out_uvs.y);

            out_col = vec4(color.rgb, 1.0 * color.a);
        }
    )";
}

std::string docscanner::frag_particle_src() {
    return version_and_precision_head R"(
        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
            float opacity = length(out_uvs - vec2(0.5));
            float alpha = step(1.0 / sqrt(2.0), 1.0 - opacity);
            out_col = vec4(1.0, 1.0, 1.0, alpha);
        }
    )";
}

std::string docscanner::frag_shutter_src() {
return version_and_precision_head R"(
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
}

std::string rounded_box_sdf_src() {
    return R"(
        float rounded_box_sdf(vec2 center_position, vec2 size, vec4 rad)
        {
            rad.xy = (center_position.x > 0.0) ? rad.xy : rad.zw;
            rad.x  = (center_position.y > 0.0) ? rad.x  : rad.y;

            vec2 q = abs(center_position)-size+rad.x;
            return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - rad.x;
        }

        float rounded_box_alpha(float distance) {
            return 1.0 - smoothstep(0.0, edge_softness, distance);
        }
    )";
}

std::string docscanner::frag_rounded_colored_quad_src() {
    return version_and_precision_head define_edge_softness R"(
        in vec2 out_rel_pos;
        in vec2 out_uvs;
        out vec4 out_col;

        uniform vec2 quad_size;
        uniform vec4 corner_rad;
        uniform vec4 light_color;
        uniform vec4 dark_color;

        const float color_mix_dist = 0.05f;
        const float color_mix_width = 0.2f;

        )" + rounded_box_sdf_src() + R"(

        void main() {
            vec2 half_size = quad_size * vec2(0.5);
            vec2 pos = quad_size * out_rel_pos - half_size;

            float distance = rounded_box_sdf(pos, half_size, corner_rad); 
            float smoothed_alpha = rounded_box_alpha(distance);

            float norm_distance = abs(distance) / length(half_size);
            float color_mix = smoothstep(color_mix_dist, color_mix_dist + color_mix_width, norm_distance);

            vec4 color = mix(dark_color, light_color, color_mix);
            out_col = vec4(color.rgb, smoothed_alpha * color.a);
        }
    )";
}

std::string docscanner::frag_rounded_textured_quad_src(bool use_oes) {
    return version_head + get_sampler_src(use_oes, 0) + precision_head + define_edge_softness + R"(
        in vec2 out_rel_pos;
        in vec2 out_uvs;
        out vec4 out_col;

        uniform float opacity;
        uniform vec2 quad_size;
        uniform vec4 corner_rad;

        )" + rounded_box_sdf_src() + R"(

        void main() {
            vec2 half_size = quad_size * vec2(0.5);
            vec2 pos = quad_size * out_rel_pos - half_size;

            float distance = rounded_box_sdf(pos, half_size, corner_rad); 
            float smoothed_alpha = rounded_box_alpha(distance);

            vec4 color = texture(sampler, out_uvs);
            out_col = vec4(color.rgb, opacity * color.a * smoothed_alpha);
        }
    )";
}

std::string docscanner::frag_sdf_quad_src() {
    return version_head "precision mediump sampler2DArray;\n" + get_array_sampler_src(0) + precision_head + R"(
        in vec2 out_rel_pos;
        in vec2 out_uvs;
        out vec4 out_col;

        uniform float zero_dist;
        uniform vec3 depths;
        uniform vec4 color;

        void main() {
            vec3 uvs_c = vec3(out_uvs, depths.x);
            vec3 uvs_n = vec3(out_uvs, depths.y);
            float sampled_dist = mix(texture(sampler, uvs_c).r, texture(sampler, uvs_n).r, fract(depths.z));
            float alpha = float(sampled_dist < zero_dist);
            out_col = vec4(color.rgb, alpha * color.a);
        }
    )";   
}