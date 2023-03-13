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
             out_col = texture(cam_sampler, out_uvs) * texture(mask_sampler, out_uvs).r;
        }
)";

const char compute_shader_cam_to_texture[] = R"(#version 320 es
        #extension GL_OES_EGL_image_external_essl3 : require
        precision mediump image2D;

        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        uniform layout(binding = 0) samplerExternalOES sampler_input;
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

const char compute_shader_cam_to_nn_input_size[] = R"(#version 320 es
        #extension GL_OES_EGL_image_external_essl3 : require
        precision mediump image2D;

        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        uniform layout(binding = 0) samplerExternalOES sampler_input;
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
    
        {{1.f, 0.f}, {1, cam_tex_left}},   // 1, 0
        {{0.f, 0.f}, {1, cam_tex_right}},  // 0, 0
        {{0.f, 1.f}, {0, cam_tex_right}}   // 0, 1
    };

    cam_quad_buffer = make_shader_buffer();
    fill_shader_buffer(cam_quad_buffer, vertices, sizeof(vertices));


    cam_to_tex_program = compile_and_link_program(compute_shader_cam_to_texture);
    ASSERT(cam_to_tex_program.program, "Cam to texture program could not be compiled.");

    nn_input_program = compile_and_link_program(compute_shader_cam_to_nn_input_size);
    ASSERT(nn_input_program.program, "NN input program could not be compiled.");

    nn_input_buffer_size = 128 * 128 * 4 * sizeof(float);
    nn_input_buffer = new u8[nn_input_buffer_size];

    nn_output_buffer_size = 128 * 128 * 1 * sizeof(float);
    nn_output_buffer = new u8[nn_output_buffer_size];
    
    cam_tex = create_texture(cam_tex_size, GL_RGBA32F);
    nn_input_tex = create_texture({128, 128}, GL_RGBA32F);
    nn_output_tex = create_texture({128, 128}, GL_R32F);

    nn_input_fb = framebuffer_from_texture(nn_input_tex, nn_input_buffer_size);
    
    nn = create_neural_network_from_path(file_ctx, "seg_model.tflite", execution_pref::sustained_speed);

    is_init = true;
}

void docscanner::cam_preview::init_cam(ANativeWindow* texture_window) {
    init_camera_capture_to_native_window(cam, texture_window);
}

void docscanner::cam_preview::render() {
    if(!is_init) return;
    
    // OES texture to gl texture
    use_program(cam_to_tex_program);

    bind_image_to_slot(0, cam_tex);
    dispatch_compute_program(cam_tex_size, 1);
        
    // downscale gl texture
    use_program(nn_input_program);

    bind_image_to_slot(0, nn_input_tex);
    dispatch_compute_program({128, 128}, 1);
    
    get_framebuffer_data(nn_input_fb, nn_input_buffer, nn_input_buffer_size);
    
    invoke_neural_network_on_data(nn, nn_input_buffer, nn_input_buffer_size, nn_output_buffer, nn_output_buffer_size);

    set_texture_data(nn_output_tex, nn_output_buffer, 128, 128);

    canvas c = {
        .bg_color={0, 1, 0}
    };
    
    use_program(preview_program);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    bind_texture_to_slot(0, nn_output_tex);
    bind_texture_to_slot(1, cam_tex);
    draw(c);
}