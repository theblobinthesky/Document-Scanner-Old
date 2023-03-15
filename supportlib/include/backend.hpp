#pragma once
#include "types.hpp"
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

NAMESPACE_BEGIN

struct shader_program {
    u32 program;
};

struct texture {
    u32 id;
    u32 format;
};

struct shader_buffer {
    u32 id;
};

struct frame_buffer {
    u32 id;
};

struct vertex {
    vec2 pos;
    vec2 uv;
};

struct canvas {
    vec3 bg_color;
};

struct variable {
    int location;

    void set_mat4(float* data);
};

struct texture_downsampler {
    uvec2 input_size, output_size;
    bool input_is_oes_texture;
    const texture* input_tex;

    texture temp_tex, output_tex;
    frame_buffer temp_fb, output_fb;
    
    shader_program gauss_blur_x_program;
    shader_program gauss_blur_y_program;

    shader_buffer gauss_quad_buffer;

    void init(uvec2 input_size, uvec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor);
    texture* downsample();
};

extern shader_program cam_preview_program;

void check_gl_error(const char* op);

shader_program compile_and_link_program(const char* vert_src, const char* frag_src, u32* vert_out, u32* frag_out);

shader_program compile_and_link_program(const char* comp_src);

void delete_program(shader_program &program);

void use_program(const shader_program &program);

void dispatch_compute_program(const uvec2 size, u32 depth);

shader_buffer make_shader_buffer();

void fill_shader_buffer(const shader_buffer& buff, vertex* vertices, u32 vertices_size, u32* indices, u32 indices_size);

texture create_texture(uvec2 size, u32 format);

void bind_image_to_slot(u32 slot, const texture &tex);

void bind_texture_to_slot(u32 slot, const texture &tex);

void bind_framebuffer(const frame_buffer &fb);

frame_buffer framebuffer_from_texture(const texture& tex);

void get_framebuffer_data(const frame_buffer &fb, const uvec2 &size, u8* &data, u32 data_size);

void set_texture_data(const texture &tex, u8* data, int width, int height);

variable get_variable(const shader_program& program, const char* name);

void draw(const canvas &canvas);

void mat4f_load_ortho(float left, float right, float bottom, float top, float near, float far, float* mat4f);

NAMESPACE_END