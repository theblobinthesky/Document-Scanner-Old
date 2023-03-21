#pragma once
#include "types.hpp"
#include "shader_program.hpp"

#ifdef ANDROID
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#elif defined(LINUX)
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#include <vector>

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
    void set_vec2(const vec2& v);
};

struct texture_downsampler {
    shader_programmer* programmer;

    uvec2 input_size, output_size;
    bool input_is_oes_texture;
    const texture* input_tex;

    texture temp_tex, output_tex;
    frame_buffer temp_fb, output_fb;
    
    shader_program gauss_blur_x_program;
    shader_program gauss_blur_y_program;

    shader_buffer gauss_quad_buffer;

    void init(shader_programmer* programmer, uvec2 input_size, uvec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor);
    texture* downsample();
};

struct sticky_particle_system {
    const std::vector<vertex>* stick_vertices;
    svec2 stick_size;

    std::vector<vertex> mesh_vertices;
    std::vector<u32> mesh_indices;

    shader_program shader;
    shader_buffer buffer;

    void gen_and_fill_mesh_vertices();
    void init(shader_programmer* programmer, const std::vector<vertex>& stick_vertices, const svec2& stick_size, shader_program shader, shader_buffer buffer);
    void render();
};

struct mesh_border {
    const std::vector<vertex>* border_vertices;
    svec2 border_size;

    std::vector<vertex> mesh_vertices;
    std::vector<u32> mesh_indices;

    shader_program shader;
    shader_buffer buffer;

    void gen_and_fill_mesh_vertices();
    void init(shader_programmer* programmer, const std::vector<vertex>& border_vertices, const svec2& border_size, shader_program shader, shader_buffer buffer);
    void render();
};

void check_gl_error(const char* op);

u32 compile_shader(u32 type, const char* src);

shader_program compile_and_link_program(u32 vert_shader, u32 frag_shader);

shader_program compile_and_link_program(u32 comp_shader);

void delete_shader(u32 id);

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