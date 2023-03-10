#include "pipeline.h"
#include "log.h"
#include "backend.h"
#include "android_camera.h"
#include "nn.h"

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
        // #extension GL_OES_EGL_image_external_essl3 : require
        precision mediump float;

        // uniform samplerExternalOES tex_sampler;
        uniform layout(binding = 0) sampler2D tex_sampler;

        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
             out_col = texture(tex_sampler, out_uvs);
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

void docscanner::pipeline::pre_init_preview(int* cam_width, int* cam_height) {
    cam = find_and_open_back_camera(cam_out_size.x, cam_out_size.y);
    *cam_width = (int) cam_out_size.x;
    *cam_height = (int) cam_out_size.y;
}

void docscanner::pipeline::init_preview(uvec2 preview_size) {
    // shader stuff
    
    preview_program = compile_and_link_program(vert_src, frag_src, nullptr, nullptr);
    ASSERT(preview_program.program, "Preview program could not be compiled.");

    use_program(preview_program);

    proj_matrix_var = get_variable(preview_program, "projection");
    
    float projection[16];
    mat4f_load_ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f, projection);

    proj_matrix_var.set_mat4(projection);


    // buffer stuff
    float p = (cam_out_size.x / (float) cam_out_size.y) * (preview_size.x / (float) preview_size.y);
    float l = (1.0f - p) / 2.0f;
    float r = 1.0f - l;
    
    vertex vertices[] = {
        {{1.f, 0.f}, {1, l}},   // 1, 0
        {{0.f, 1.f}, {0, r}},   // 0, 1
        {{1.f, 1.f}, {0, l}},   // 1, 1
    
        {{1.f, 0.f}, {1, l}},   // 1, 0
        {{0.f, 0.f}, {1, r}},   // 0, 0
        {{0.f, 1.f}, {0, r}}    // 0, 1
    };

    vbo = make_shader_buffer();
    fill_shader_buffer(vbo, vertices, sizeof(vertices));
}


void docscanner::pipeline::init_cam(ANativeWindow* texture_window, file_context* file_ctx) {
    // camera stuff
    init_camera_capture_to_native_window(cam, texture_window);

    nn_input_program = compile_and_link_program(compute_shader_cam_to_nn_input_size);
    ASSERT(nn_input_program.program, "NN input program could not be compiled.");

    nn_input_buffer_size = 128 * 128 * 4 * sizeof(float);
    nn_input_buffer = new u8[nn_input_buffer_size];

    nn_output_buffer_size = 128 * 128 * 1 * sizeof(float);
    nn_output_buffer = new u8[nn_output_buffer_size];
    
    for(int i = 0; i < nn_output_buffer_size; i++) {
        if(i % 100 < 50) nn_output_buffer[i] = 0;
        else nn_output_buffer[i] = 100;
    }

    nn_input_tex = create_texture({128, 128}, GL_RGBA32F);
    nn_output_tex = create_texture({128, 128}, GL_R32F);

    nn_input_fb = framebuffer_from_texture(nn_input_tex, nn_input_buffer_size);
    
    nn = create_neural_network_from_path(file_ctx, "seg_model.tflite", execution_pref::sustained_speed);
}
    
void docscanner::pipeline::render_preview() {
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
    draw(c);
}