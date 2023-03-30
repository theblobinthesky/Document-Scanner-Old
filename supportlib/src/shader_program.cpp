#include "shader_program.hpp"
#include "log.hpp"
#include "backend.hpp"
#include <functional>
#include <math.h>

using namespace docscanner;

std::string docscanner::frag_simple_tex_sampler_src(bool oes_input, u32 binding_slot) {
    std::string sampler_src = oes_input ? 
    R"( 
        #extension GL_OES_EGL_image_external_essl3 : require

        uniform layout(binding = )" + std::to_string(binding_slot) + R"() samplerExternalOES cam_sampler;
    )" : 
    R"( 
        uniform layout(binding = )" + std::to_string(binding_slot) + R"() sampler2D cam_sampler;
    )";

    return version_head + sampler_src + R"(
        precision mediump float;
        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
            out_col = texture(cam_sampler, out_uvs);
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
    std::string sampler_src = use_oes ? 
    R"( 
        #extension GL_OES_EGL_image_external_essl3 : require
        uniform layout(binding = 0) samplerExternalOES sampler;
    )" : 
    R"(
        uniform layout(binding = 0) sampler2D sampler;
    )";

    u32 M = N / 2 + 1;

    return version_head + sampler_src + R"(
        precision mediump float;
        
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