#include "pipeline.h"
#include "log.h"
#include "backend.h"
#include "android_camera.h"

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
        #extension GL_OES_EGL_image_external_essl3 : require
        precision mediump float;

        uniform samplerExternalOES tex_sampler;

        in vec2 out_uvs;
        out vec4 out_col;

        void main() {
             out_col = texture(tex_sampler, out_uvs);
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
    if (!preview_program.program) {
        LOGE_AND_BREAK("Preview program could not be compiled.");
        return;
    }

    use_program(preview_program);

    proj_matrix_var = get_variable(preview_program, "projection");
    
    float projection[16];
    mat4f_load_ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f, projection);

    proj_matrix_var.set_mat4(projection);


    // buffer stuff
    float p = (cam_out_size.x / (float) cam_out_size.y) * (preview_size.x / (float) preview_size.y);
    float l = (1.0f - p) / 2.0f;
    float r = 1.0f - l;
    LOGI("l: %f, r: %f", l, r);
    
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


void docscanner::pipeline::init_cam(ANativeWindow* texture_window) {
    // camera stuff
    init_camera_capture_to_native_window(cam, texture_window);
}
    
void docscanner::pipeline::render_preview() {
    canvas c = {
        .bg_color={0, 1, 0}
    };
    
    draw(c);
}