#include "cam_preview.h"
#include "log.h"
#include "backend.h"
#include "android_camera.h"

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
             out_col = texture(cam_sampler, out_uvs); // * texture(mask_sampler, out_uvs).r;
        }
)";

constexpr const char gauss_blur_frag_src[] = R"(#version 310 es
#if %d
        #extension GL_OES_EGL_image_external_essl3 : require
        uniform layout(binding = 0) samplerExternalOES sampler;
#else
        uniform layout(binding = 0) sampler2D sampler;
#endif

        precision mediump float;
        
        in vec2 out_uvs;
        out vec4 out_col;

        const int M = %u;
        const int N = 2 * M + 1;
        const float coeffs[N] = float[N](%s);
        const vec2 pixel_shift = vec2(%f, %f);

        void main() {
            vec4 col = vec4(0);
            for(int i = 0; i < N; i++) {
                col += coeffs[i] * texture(sampler, out_uvs + float(i - M) * pixel_shift);
            }

            out_col = col;
        }
)";

const char compute_shader_cam_to_nn_input_size[] = R"(#version 320 es
        precision mediump image2D;

        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        uniform layout(binding = 0) sampler2D sampler_input;
        uniform layout(binding = 0, rgba32f) writeonly image2D img_output;

        void main() {
            ivec2 pt_xy = ivec2(gl_GlobalInvocationID.xy);
            vec2 pt_uv = vec2(
                float(gl_GlobalInvocationID.x) / float(gl_NumWorkGroups.x - 1u),
                float(gl_GlobalInvocationID.y) / float(gl_NumWorkGroups.y - 1u)
            );

            vec4 color = texture(sampler_input, pt_uv);
            imageStore(img_output, pt_xy, color);
        }
)";

void docscanner::cam_preview::pre_init(int* cam_width, int* cam_height) {
    cam = find_and_open_back_camera(cam_tex_size.x, cam_tex_size.y);
    *cam_width = (int) cam_tex_size.x;
    *cam_height = (int) cam_tex_size.y;
}

#include <string>
#include <math.h>

std::string compute_gauss_coefficients(int n) {
    f32 sigma = (n - 1.0f) / 2.0f;
    
    f32 *coeffs = new f32[n];
    f32 sum = 0.0f;

    for(int i = 0; i < n; i++) {
        const f32 gauss_factor = 1.0 / sqrt(2 * M_PI * sigma);
        const f32 x = (f32)(i - n / 2);
        coeffs[i] = gauss_factor * pow(M_E, -(x * x) / (2 * sigma * sigma));
        sum += coeffs[i];
    } 

    // normalize
    for(int i = 0; i < n; i++) {
        coeffs[i] /= sum;
    }

    std::string ret = "";
    for(int i = 0; i < n; i++) {
        ret = ret + std::to_string(coeffs[i]);

        if(i != n - 1) ret = ret + ", ";
    }

    delete[] coeffs;

    return ret;
}

char* prepare_gauss_fragment_src(bool sample_from_external, u32 n, vec2 pixel_shift) {
    std::string gauss_coeffs = compute_gauss_coefficients(n);

    size_t needed = snprintf(null, 0, gauss_blur_frag_src, sample_from_external, n / 2, gauss_coeffs.c_str(), pixel_shift.x, pixel_shift.y);
    
    char* buff = new char[needed];
    sprintf(buff, gauss_blur_frag_src, sample_from_external, n / 2, gauss_coeffs.c_str(), pixel_shift.x, pixel_shift.y);
    return buff;
}

void docscanner::cam_preview::init_cam_stuff() {
#define EVEN_TO_UNEVEN(n) if ((n) % 2 == 0) { n++; }

    uvec2 req_kernel_size = {
        (u32)(cam_tex_size.x / 128.0),
        (u32)(cam_tex_size.y / 128.0)
    };

    EVEN_TO_UNEVEN(req_kernel_size.x);
    EVEN_TO_UNEVEN(req_kernel_size.y);

#undef EVEN_TO_UNEVEN

    cam_tex = create_texture({128, 128}, GL_RGBA16F);
    cam_tex_2 = create_texture({128, cam_tex_size.y}, GL_RGBA16F);

    cam_fb = framebuffer_from_texture(cam_tex);
    cam_fb_2 = framebuffer_from_texture(cam_tex_2);

    char* gauss_frag_src_x = prepare_gauss_fragment_src(true, req_kernel_size.x, {1.0f / (f32)cam_tex_size.x, 0.0f});
    LOGI("%s", gauss_frag_src_x);
    gauss_blur_x_program = compile_and_link_program(vert_src, gauss_frag_src_x, null, null);
    ASSERT(gauss_blur_x_program.program, "gauss_blur_x_program program could not be compiled.");
    
    char* gauss_frag_src_y = prepare_gauss_fragment_src(false, req_kernel_size.y, {0.0f, 1.0f / (f32)cam_tex_size.y});
    gauss_blur_y_program = compile_and_link_program(vert_src, gauss_frag_src_y, null, null);
    ASSERT(gauss_blur_y_program.program, "gauss_blur_y_program program could not be compiled.");

    float projection[16];
    mat4f_load_ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f, projection);

    use_program(gauss_blur_x_program);
    auto proj_matrix_x_var = get_variable(gauss_blur_x_program, "projection");
    proj_matrix_x_var.set_mat4(projection);

    use_program(gauss_blur_y_program);
    auto proj_matrix_y_var = get_variable(gauss_blur_y_program, "projection");    
    proj_matrix_y_var.set_mat4(projection);

    vertex vertices[] = {
        {{1.f, 0.f}, {1, 0}},
        {{0.f, 1.f}, {0, 1}},
        {{1.f, 1.f}, {1, 1}},
        {{0.f, 0.f}, {0, 0}}
    };

    u32 indices[] = { 
        0, 1, 2, 
        0, 3, 1 
    };

    gauss_quad_buffer = make_shader_buffer();
    fill_shader_buffer(gauss_quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));
}

void docscanner::cam_preview::init_backend(uvec2 preview_size, file_context* file_ctx) {
    preview_program = compile_and_link_program(vert_src, frag_src, nullptr, nullptr);
    ASSERT(preview_program.program, "Preview program could not be compiled.");

    use_program(preview_program);

    auto proj_matrix_var = get_variable(preview_program, "projection");
    
    float projection[16];
    mat4f_load_ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f, projection);

    proj_matrix_var.set_mat4(projection);


    // buffer stuff
    float p = (cam_tex_size.x / (float) cam_tex_size.y) * (preview_size.x / (float) preview_size.y);
    cam_tex_left = (1.0f - p) / 2.0f;
    cam_tex_right = 1.0f - cam_tex_left;
    
    vertex vertices[] = {
        {{1.f, 0.f}, {1, cam_tex_left}},   // 1, 0
        {{0.f, 1.f}, {0, cam_tex_right}},  // 0, 1
        {{1.f, 1.f}, {0, cam_tex_left}},   // 1, 1
        {{0.f, 0.f}, {1, cam_tex_right}}  // 0, 0
    };

    u32 indices[] = { 
        0, 1, 2, 
        0, 3, 1 
    };

    cam_quad_buffer = make_shader_buffer();
    fill_shader_buffer(cam_quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));


    nn_input_program = compile_and_link_program(compute_shader_cam_to_nn_input_size);
    ASSERT(nn_input_program.program, "NN input program could not be compiled.");

    nn_input_buffer_size = 128 * 128 * 4 * sizeof(float);
    nn_input_buffer = new u8[nn_input_buffer_size];

    nn_output_buffer_size = 128 * 128 * 1 * sizeof(float);
    nn_output_buffer = new u8[nn_output_buffer_size];
    
    nn_input_tex = create_texture({128, 128}, GL_RGBA32F);
    nn_output_tex = create_texture({128, 128}, GL_R32F);

    nn_input_fb = framebuffer_from_texture(nn_input_tex);
    
    /*
    nn = create_neural_network_from_path(file_ctx, "seg_model.tflite", execution_pref::sustained_speed);
    */

    init_cam_stuff();
}

void docscanner::cam_preview::init_cam(ANativeWindow* texture_window) {
    init_camera_capture_to_native_window(cam, texture_window);
}

#include <chrono>

void docscanner::cam_preview::render() {
    auto start = std::chrono::high_resolution_clock::now();
    
    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    // first blur pass    
    use_program(gauss_blur_x_program);

    glBindFramebuffer(GL_FRAMEBUFFER, cam_fb_2);
    glBindVertexArray(gauss_quad_buffer.id);

    glViewport(0, 0, 128, cam_tex_size.y);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");


    // second blur pass
    use_program(gauss_blur_y_program);
    bind_texture_to_slot(0, cam_tex_2);
    
    glBindFramebuffer(GL_FRAMEBUFFER, cam_fb);
    glBindVertexArray(gauss_quad_buffer.id);

    glViewport(0, 0, 128, 128);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");


    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

    // downscale gl texture
    use_program(nn_input_program);

    bind_texture_to_slot(0, cam_tex);
    bind_image_to_slot(0, nn_input_tex);
    dispatch_compute_program({128, 128}, 1);

    glFinish();

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOGI("gl downscaling takes %lldms", dur.count());

    /*    
    get_framebuffer_data(nn_input_fb, nn_input_buffer, nn_input_buffer_size);
    
    invoke_neural_network_on_data(nn, nn_input_buffer, nn_input_buffer_size, nn_output_buffer, nn_output_buffer_size);

    set_texture_data(nn_output_tex, nn_output_buffer, 128, 128);
    */

    canvas c = {
        .bg_color={0, 1, 0}
    };
    
    use_program(preview_program);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(cam_quad_buffer.id);
    
    bind_texture_to_slot(0, nn_output_tex);
    bind_texture_to_slot(1, cam_tex);
    draw(c);
}