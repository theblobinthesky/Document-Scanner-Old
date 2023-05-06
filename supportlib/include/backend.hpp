#pragma once
#include "platform.hpp"
#include "shader_program.hpp"

#include <vector>
#include <queue>
#include <stack>

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
    svec2 size;
};

struct stack_texture {
    u32 id;
    u32 format;
    svec2 size;
    s32 depth;
};

struct shader_buffer {
    u32 vao, vbo, ebo;
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

struct variable {
    int location;

    void set_f32(f32 v);
    void set_mat4(const mat4& mat);
    void set_vec2(const vec2& v);
    void set_vec3(const vec3& v);
    void set_vec4(const rect& r);
    void set_vec4(const vec2& a, const vec2& b);
    void set_vec4(const vec4& v);
    void set_bool(bool v);
};

struct instanced_quad {
    vec2 v0, v1, v2, v3;
    vec2 uv_tl, uv_br;
};

struct instanced_quads {
    instanced_quad* quads;
    s32 quads_size;
    instanced_shader_buffer quads_buffer;

    void init(s32 size);
    void fill();
    void draw();
};

#ifdef DEBUG
struct DEBUG_marker {
    vec2 pos;
    vec3 color;
};
#endif

enum class rot_mode {
    ROT_0_DEG,
    ROT_90_DEG,
    ROT_180_DEG,
    ROT_270_DEG
};

struct composite_group {
    vec3 bg_color;
    bool bg_transparent;
    f32 opacity;
};

struct engine_backend;

struct scoped_composite_group {
    engine_backend* backend;

    scoped_composite_group(engine_backend* backend, vec3 bg_color, bool bg_transparent, f32 opacity);
    ~scoped_composite_group();
};

#define SCOPED_COMPOSITE_GROUP(backend, bg_color, bg_transparent, opacity) \
    scoped_composite_group SCOPED_COMPOSITE_GROUP_INSTANCE(backend, bg_color, bg_transparent, opacity);

struct scoped_transform {
    engine_backend* backend;

    scoped_transform(engine_backend* backend, vec2 transform);
    ~scoped_transform();
};

#define SCOPED_TRANSFORM(backend, transform) \
    scoped_transform SCOPED_TRANSFORM_INSTANCE(backend, transform);

struct colored_quad_desc {
    rect bounds;
    rect crad = {};
    vec4 color;
    vec4 border_color    = vec4::fill_value(0);
    f32 border_thickness = 0;
};

struct textured_quad_desc {
    rect bounds;
    rect crad = {};
    texture tex;
    f32 opacity = 1.0f;
    rect uv_bounds = { {}, {1, 1} };
    rot_mode uv_rot = rot_mode::ROT_0_DEG;
    vec4 border_color    = vec4::fill_value(0);
    f32 border_thickness = 0;
};

struct sdf_quad_desc {
    rect bounds;
    stack_texture tex;
    vec4 color;
    f32 from_depth = 0.0f;
    f32 to_depth = 0.0f;
    f32 blend_depth = 0.0f;
    f32 blendin;
    rot_mode uv_rot = rot_mode::ROT_0_DEG;
};

struct engine_backend {
    input_manager input;
    file_context* file_ctx;
    thread_pool* threads;

    camera* cam;
    bool cam_is_init;
    svec2 cam_size_px;

    std::unordered_map<u64, u32> shader_map;
    std::unordered_map<u64, u32> program_map;
    mat4 projection_mat;
    
    shader_buffer quad_buffer;
    shader_program rounded_quad_with_color;
    shader_program rounded_quad_with_texture;
    shader_program sdf_quad_desc_with_texture;

#ifdef USES_OES_TEXTURES
    shader_program rounded_quad_with_oes_texture;
#endif

    svec2 preview_size_px;
    f32 preview_height;

    composite_group comp;
    texture comp_texture;
    frame_buffer comp_fb;

    std::stack<vec2> transform_stack;
    vec2 global_transform;

#ifdef DEBUG
    shader_program DEBUG_marker_program;
    std::vector<DEBUG_marker> DEBUG_marker_queue;
#endif

    f32 time;

    engine_backend(file_context* file_ctx, thread_pool* threads, svec2 preview_size_px);
    void init_camera_related(camera* cam, svec2 cam_size_px);

    shader_program compile_and_link(const std::string& vert_src, const std::string& frag_src);
    shader_program compile_and_link(const std::string& comp_src);
    void use_program(const shader_program& program);

    void clear_screen(vec3 color);
    void begin_composite_group(const composite_group& comp);
    void end_composite_group();

    void draw_quad(const shader_program& program, const rect& bounds);
    void draw_quad(const shader_program& program, const rect& bounds, const rect& uv_bounds);
    void draw_quad(const shader_program& program, const rect& bounds, const rect& uv_bounds, rot_mode uv_rot);
    void draw_rounded_colored_quad_desc(const colored_quad_desc& quad);
    void draw_rounded_textured_quad_desc(const textured_quad_desc& quad);
    void draw_colored_sdf_quad_desc(const sdf_quad_desc& quad);

#ifdef USES_OES_TEXTURES
    void draw_rounded_oes_textured_quad_desc(const textured_quad_desc& quad);
#endif

#ifdef DEBUG
    void DEBUG_draw_marker(const vec2& pt, const vec3& col);
    void DEBUG_draw();
#endif
};

struct scoped_camera_matrix {
    engine_backend* backend;
    mat4 previous_matrix;

    scoped_camera_matrix(engine_backend* backend, const mat4& camera_matrix);
    ~scoped_camera_matrix();
};

#define SCOPED_CAMERA_MATRIX(backend, matrix) scoped_camera_matrix SCOPED_CAMERA_MATRIX_VAR((backend), (matrix))

struct texture_sampler {
    engine_backend* backend;
    mat4 projection_matrix;

    bool input_is_oes_texture;
    const texture* input_tex;

    svec2 output_size;
    texture output_tex;
    frame_buffer output_fb;
    
    const vertex* vertices;
    u32 vertices_size;
    const u32* indices;
    u32 indices_size;
    shader_buffer buffer;

    shader_program sampler_program;
    
    void init(engine_backend* backend, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, 
              const vertex* vertices, u32 vertices_size, const u32* indices, u32 indices_size);
    void sample();
};

struct texture_downsampler_stage {
    engine_backend* backend;

    svec2 input_size;
    svec2 output_size;
    bool input_is_oes_texture;
    const texture* input_tex;

    texture temp_tex, output_tex;
    frame_buffer temp_fb, output_fb;
    
    shader_program gauss_blur_x_program;
    shader_program gauss_blur_y_program;
    
    void init(engine_backend* backend, svec2 input_size, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor);
    void downsample();
};

struct texture_downsampler {
    engine_backend* backend;
    mat4 projection_matrix;

    svec2 input_size;
    svec2 output_size;
    bool input_is_oes_texture;
    const texture* input_tex;
    const texture* output_tex;
    const frame_buffer* output_fb;

    texture_downsampler_stage* stages;
    s32 stages_size;

    shader_buffer gauss_quad_buffer;

    GLsync fence;

    void init(engine_backend* backend, svec2 input_size, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, s32 downsampling_stages, f32 relaxation_factor);
    void downsample();
};

// Line rendering inspired by
// https://wwwtyro.net/2019/11/18/instanced-lines.html

struct lines {
    engine_backend* backend;

    vec2* points;
    bool is_closed;
    vec2* closed_points;
    s32 points_size;
    f32 thickness;
    vec4 color;

    instanced_shader_buffer lines_buffer;
    instanced_shader_buffer joins_buffer;

    shader_program lines_program;
    shader_program joins_program;

    void init(engine_backend* backend, vec2* points, s32 points_size, f32 thickness, vec3 color, bool is_closed);
    void fill();
    void draw();
};

void check_gl_error(const char* op);

u32 compile_shader(u32 type, const char* src);

shader_program compile_and_link_program(u32 vert_shader, u32 frag_shader);

shader_program compile_and_link_program(u32 comp_shader);

bool load_program_from_binary(file_context* ctx, u64 hash, shader_program& program);

void save_program_to_binary(file_context* ctx, u64 hash, shader_program& program);

void delete_shader(u32 id);

void delete_program(shader_program &program);

void dispatch_compute_program(const svec2 size, u32 depth);

shader_buffer make_shader_buffer();

instanced_shader_buffer make_instanced_quad_shader_buffer(shader_buffer buff);

instanced_shader_buffer make_instanced_point_shader_buffer(shader_buffer buff);

instanced_shader_buffer make_instanced_line_shader_buffer(shader_buffer buff);

void bind_shader_buffer(const shader_buffer& buff);

void fill_shader_buffer(const shader_buffer& buff, const vertex* vertices, u32 vertices_size, const u32* indices, u32 indices_size);

texture make_texture(svec2 size, u32 format);

stack_texture make_stack_texture(s32 depth, svec2 size, u32 format);

void bind_image_to_slot(u32 slot, const texture &tex);

void bind_texture_to_slot(u32 slot, const texture &tex);

void bind_texture_to_slot(u32 slot, const stack_texture &tex);

void bind_framebuffer(const frame_buffer &fb);

void unbind_framebuffer();

frame_buffer framebuffer_from_texture(const texture& tex);

void get_framebuffer_data(const frame_buffer &fb, const svec2 &size, u8* &data, u32 data_size);

void set_texture_data(const texture &tex, u8* data, const svec2& size);

void set_texture_data(const stack_texture &tex, u8* data, const svec2& size);

variable get_variable(const shader_program& program, const char* name);

NAMESPACE_END