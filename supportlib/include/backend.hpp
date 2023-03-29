#pragma once
#include "types.hpp"
#include "shader_program.hpp"
#include <vector>

// nocheckin
#ifdef ANDROID
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#elif defined(LINUX)
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#define DEBUG

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

struct instanced_shader_buffer {
    u32 vao;
    u32 instance_vbo;
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

    void set_f32(f32 v);
    void set_mat4(const mat4& mat);
    void set_vec2(const vec2& v);
    void set_vec3(const vec3& v);
    void set_vec4(const vec2& a, const vec2& b);
};

struct instanced_quad {
    vec2 v0, v1, v2, v3;
};

struct instanced_quads {
    instanced_quad* quads;
    s32 quads_size;
    instanced_shader_buffer quads_buffer;

    void init(s32 size);
    void fill();
    void draw();
};

struct lines {
    vec2* points;
    vec2* closed_points;
    s32 points_size;
    instanced_shader_buffer lines_buffer;

    void init(vec2* points, s32 points_size);
    void fill();
    void draw();
};

#ifdef DEBUG
struct DEBUG_marker {
    vec2 pos;
    vec3 color;
};
#endif

struct engine_backend {
    std::unordered_map<u64, u32> shader_map;
    std::unordered_map<u64, u32> program_map;
    mat4 projection_mat;
    
    shader_buffer quad_buffer;

#ifdef DEBUG
    shader_program DEBUG_marker_program;
    std::vector<DEBUG_marker> DEBUG_marker_queue;
#endif

    f32 time;

    void init(mat4 projection_mat);

    shader_program compile_and_link(const std::string& vert_src, const std::string& frag_src);
    shader_program compile_and_link(const std::string& comp_src);

    void draw_quad(const vec2& pos, const vec2& size);

#ifdef DEBUG
    void DEBUG_draw_marker(const vec2& pt, const vec3& col);
    void DEBUG_draw();
#endif
};

struct texture_downsampler {
    uvec2 input_size;
    svec2 output_size;
    bool input_is_oes_texture;
    const texture* input_tex;

    texture temp_tex, output_tex;
    frame_buffer temp_fb, output_fb;
    
    shader_program gauss_blur_x_program;
    shader_program gauss_blur_y_program;

    shader_buffer gauss_quad_buffer;

    void init(engine_backend* backend, uvec2 input_size, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor);
    texture* downsample();
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

instanced_shader_buffer make_instanced_quad_shader_buffer(shader_buffer buff);

instanced_shader_buffer make_instances_lines_shader_buffer(shader_buffer buff);

void bind_shader_buffer(const shader_buffer& buff);

void fill_shader_buffer(const shader_buffer& buff, vertex* vertices, u32 vertices_size, u32* indices, u32 indices_size);

texture create_texture(uvec2 size, u32 format);

void bind_image_to_slot(u32 slot, const texture &tex);

void bind_texture_to_slot(u32 slot, const texture &tex);

void bind_framebuffer(const frame_buffer &fb);

void unbind_framebuffer();

frame_buffer framebuffer_from_texture(const texture& tex);

void get_framebuffer_data(const frame_buffer &fb, const svec2 &size, u8* &data, u32 data_size);

void set_texture_data(const texture &tex, u8* data, int width, int height);

variable get_variable(const shader_program& program, const char* name);

void draw(const canvas &canvas);

void mat4f_load_ortho(float left, float right, float bottom, float top, float near, float far, float* mat4f);

NAMESPACE_END