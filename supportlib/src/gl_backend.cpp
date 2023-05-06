#if 1
#include "backend.hpp"
#include <string.h>
#include <math.h>

#if defined(LINUX)
#include <cstddef>
#endif

#ifdef ANDROID
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#elif defined(LINUX)
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

using namespace docscanner;

constexpr vec2 DEBUG_marker_size = { 0.02f, 0.02f };
constexpr s32 vertex_attrib_count = 2;

void docscanner::check_gl_error(const char* op) {
    for (GLenum error = glGetError(); error; error = glGetError()) {
        LOGE_AND_BREAK("glError with code 0x%04x was triggered by %s().\n", error, op);
    }
}

void docscanner::variable::set_f32(f32 v) {
    glUniform1f(location, v);
}

void docscanner::variable::set_mat4(const mat4& mat) {
    glUniformMatrix4fv(location, 1, GL_FALSE, mat.data);
}

void docscanner::variable::set_vec2(const vec2& v) {
    glUniform2f(location, v.x, v.y);
}

void docscanner::variable::set_vec3(const vec3& v) {
    glUniform3f(location, v.x, v.y, v.z);
}

void docscanner::variable::set_vec4(const rect& r) {
    glUniform4f(location, r.tl.x, r.tl.y, r.br.x, r.br.y);
}

void docscanner::variable::set_vec4(const vec2& a, const vec2& b) {
    glUniform4f(location, a.x, a.y, b.x, b.y);
}

void docscanner::variable::set_vec4(const vec4& v) {
    glUniform4f(location, v.x, v.y, v.z, v.w);
}

void docscanner::variable::set_bool(bool v) {
    glUniform1i(location, v);
}

void docscanner::instanced_quads::init(s32 size) {
    quads = new instanced_quad[size];
    quads_size = size;
    
    u32 indices[6] = { 
        0, 1, 2, 
        0, 2, 3 
    };

    vertex vertices[5] = {
        {{0, 0}, {0, 0}},
        {{1, 0}, {1, 0}},
        {{1, 1}, {1, 1}},
        {{0, 1}, {0, 1}}
    };

    shader_buffer quad_buffer = make_shader_buffer();
    fill_shader_buffer(quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    quads_buffer = make_instanced_quad_shader_buffer(quad_buffer);

    fill();
}

void docscanner::instanced_quads::fill() {
    glBindBuffer(GL_ARRAY_BUFFER, quads_buffer.instance_vbo);
    glBufferData(GL_ARRAY_BUFFER, quads_size * sizeof(instanced_quad), quads, GL_DYNAMIC_DRAW);
}

void docscanner::instanced_quads::draw() {
    glBindVertexArray(quads_buffer.vao);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null, quads_size);
}

scoped_composite_group::scoped_composite_group(engine_backend* backend, vec3 bg_color, bool bg_transparent, f32 opacity) : backend(backend) {
    composite_group comp = {
        .bg_color = bg_color, 
        .bg_transparent = bg_transparent,
        .opacity = opacity
    };

    backend->begin_composite_group(comp);
}

scoped_composite_group::~scoped_composite_group() {
    backend->end_composite_group();
}

scoped_transform::scoped_transform(engine_backend* backend, vec2 transform) : backend(backend) {
    backend->transform_stack.push(transform);
    backend->global_transform = backend->global_transform + transform;
}

scoped_transform::~scoped_transform() {
    vec2 transform = backend->transform_stack.top();
    backend->transform_stack.pop();
    backend->global_transform = backend->global_transform - transform;
}

engine_backend::engine_backend(file_context* file_ctx, thread_pool* threads, svec2 preview_size_px) 
    : file_ctx(file_ctx), threads(threads), cam_is_init(false), preview_size_px(preview_size_px), global_transform({}) {
    preview_height = preview_size_px.y / (f32)preview_size_px.x;
    input.init(preview_size_px, preview_height);

    glEnable(GL_BLEND);  
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    check_gl_error("glEnable");

    u32 indices[6] = { 
        0, 1, 2, 
        0, 2, 3 
    };

    vertex vertices[5] = {
        {{0, 0}, {0, 0}},
        {{1, 0}, {1, 0}},
        {{1, 1}, {1, 1}},
        {{0, 1}, {0, 1}}
    };

    quad_buffer = make_shader_buffer();
    fill_shader_buffer(quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    rounded_quad_with_color = compile_and_link(vert_quad_src(), frag_rounded_colored_quad_src());
    rounded_quad_with_texture = compile_and_link(vert_quad_src(), frag_rounded_textured_quad_src(false));
    sdf_quad_with_texture = compile_and_link(vert_quad_src(), frag_sdf_quad_src());

#if USES_OES_TEXTURES
    rounded_quad_with_oes_texture = compile_and_link(vert_quad_src(), frag_rounded_textured_quad_src(true));
#endif
 

#ifdef DEBUG
    DEBUG_marker_program = compile_and_link(vert_quad_src(), frag_DEBUG_marker_src());
#endif

    comp_texture = make_texture(preview_size_px, GL_RGBA32F);
    comp_fb = framebuffer_from_texture(comp_texture);
}

void engine_backend::init_camera_related(camera* cam, svec2 cam_size_px) {
    this->cam = cam;
    this->cam_size_px = cam_size_px;
}

u32 find_or_insert_shader(std::unordered_map<u64, u32>& map, u32 type, const std::string& src) {   
    u64 hash = (u64)std::hash<std::string>()(src);
    auto found = map.find(hash);

    if(true || found == map.end()) {
        u32 shader = compile_shader(type, src.c_str());
        map.emplace(hash, shader);
        return shader;
    } else {
        return found->second;
    }
}

shader_program engine_backend::compile_and_link(const std::string& vert_src, const std::string& frag_src) {
    u64 program_id = std::hash<std::string>()(vert_src) ^ std::hash<std::string>()(frag_src);

    auto program_found = program_map.find(program_id);
    if(program_found == program_map.end()) {
        shader_program program;

        if(!load_program_from_binary(file_ctx, program_id, program)) {
            LOGI("recompiling shader program");

            u32 vert_shader = find_or_insert_shader(shader_map, GL_VERTEX_SHADER, vert_src);
            u32 frag_shader = find_or_insert_shader(shader_map, GL_FRAGMENT_SHADER, frag_src);
            program = compile_and_link_program(vert_shader, frag_shader);

            save_program_to_binary(file_ctx, program_id, program);
        } else {
            LOGI("cached shader program");
        }

        program_map.emplace(program_id, program.program);
        return program;
    }

    return { program_found->second };
}


shader_program engine_backend::compile_and_link(const std::string& comp_src) {
    u32 comp_shader = find_or_insert_shader(shader_map, GL_COMPUTE_SHADER, comp_src);

    u64 program_id = (u64)comp_shader;
    auto program_found = program_map.find(program_id);
    if(program_found == program_map.end()) {
        shader_program program = compile_and_link_program(comp_shader);
        program_map.emplace(program_id, program.program);
        return program;
    } else {
        return {program_found->second};
    }
}

void engine_backend::use_program(const shader_program &program) {
    glUseProgram(program.program);
    check_gl_error("glUseProgram");
    
    get_variable(program, "projection").set_mat4(projection_mat);
}

void engine_backend::clear_screen(vec3 color) {
    glClearColor(color.x, color.y, color.z, 1);
    glClear(GL_COLOR_BUFFER_BIT);
}

void engine_backend::begin_composite_group(const composite_group& comp) {
    this->comp = comp;

    bind_framebuffer(comp_fb);
    
    if(comp.bg_transparent) glClearColor(0, 0, 0, 0);
    else glClearColor(comp.bg_color.x, comp.bg_color.y, comp.bg_color.z, 1);

    glClear(GL_COLOR_BUFFER_BIT);
}

void engine_backend::end_composite_group() {
    unbind_framebuffer();

    draw_rounded_textured_quad({ .bounds = { {}, {1.0f, preview_height} }, .tex = comp_texture, .opacity = comp.opacity, .uv_rot = rot_mode::ROT_180_DEG });
}

void engine_backend::draw_quad(const shader_program& program, const rect& bounds) {
    draw_quad(program, bounds, rect({ {}, { 1, 1 } }), rot_mode::ROT_0_DEG);
}

void engine_backend::draw_quad(const shader_program& program, const rect& bounds, const rect& uv_bounds) {
    draw_quad(program, bounds, uv_bounds, rot_mode::ROT_0_DEG);
}

void engine_backend::draw_quad(const shader_program& program, const rect& bounds, const rect& uv_bounds, rot_mode uv_rot) {
    bind_shader_buffer(quad_buffer);

    rect real_bounds = bounds;
    real_bounds.tl = real_bounds.tl + global_transform;
    real_bounds.br = real_bounds.br + global_transform;

    use_program(program);
    get_variable(program, "bounds").set_vec4(real_bounds);

    bool rot_uv_90_deg = false;
    rect real_uv_bounds = uv_bounds;

    if(uv_rot == rot_mode::ROT_90_DEG || uv_rot == rot_mode::ROT_270_DEG) {
        rot_uv_90_deg = true;
        real_uv_bounds = { { uv_bounds.tl.y, uv_bounds.tl.x }, { uv_bounds.br.y, uv_bounds.br.x } };
    }

    if(uv_rot == rot_mode::ROT_180_DEG || uv_rot == rot_mode::ROT_270_DEG) {
        f32 temp_tl_y = real_uv_bounds.tl.y;
        real_uv_bounds.tl.y = real_uv_bounds.br.y;
        real_uv_bounds.br.y = temp_tl_y;
    }

    get_variable(program, "uv_bounds").set_vec4(real_uv_bounds);
    get_variable(program, "rot_uv_90_deg").set_bool(rot_uv_90_deg);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");
}


void engine_backend::draw_rounded_colored_quad(const colored_quad& quad) {
    use_program(rounded_quad_with_color);
    get_variable(rounded_quad_with_color, "quad_size").set_vec2(quad.bounds.size());
    get_variable(rounded_quad_with_color, "corner_rad").set_vec4(quad.crad);
    get_variable(rounded_quad_with_color, "light_color").set_vec4(quad.color);
    get_variable(rounded_quad_with_color, "dark_color").set_vec4(quad.color * 0.95f);
    draw_quad(rounded_quad_with_color, quad.bounds);
}

void engine_backend::draw_rounded_textured_quad(const textured_quad& quad) {
    bind_texture_to_slot(0, quad.tex);

    use_program(rounded_quad_with_texture);
    get_variable(rounded_quad_with_texture, "opacity").set_f32(quad.opacity);
    get_variable(rounded_quad_with_texture, "quad_size").set_vec2(quad.bounds.size());
    get_variable(rounded_quad_with_texture, "corner_rad").set_vec4(quad.crad);
    draw_quad(rounded_quad_with_texture, quad.bounds, quad.uv_bounds, quad.uv_rot);
}

void engine_backend::draw_colored_sdf_quad(const sdf_quad& quad) {
    bind_texture_to_slot(0, quad.tex);

    use_program(sdf_quad_with_texture);
    get_variable(sdf_quad_with_texture, "zero_dist").set_f32(quad.blendin);
    get_variable(sdf_quad_with_texture, "depths").set_vec3({quad.from_depth, quad.to_depth, quad.blend_depth * 0.999f}); // 0.999f makes it not reset at 1.0f
    get_variable(sdf_quad_with_texture, "color").set_vec4(quad.color);
    draw_quad(sdf_quad_with_texture, quad.bounds, { {}, {1, 1} }, quad.uv_rot);
}

#ifdef USES_OES_TEXTURES
void engine_backend::draw_rounded_oes_textured_quad(const textured_quad& quad) {
    use_program(rounded_quad_with_oes_texture);
    get_variable(rounded_quad_with_texture, "opacity").set_f32(1.0f);
    get_variable(rounded_quad_with_oes_texture, "quad_size").set_vec2(quad.bounds.size());
    get_variable(rounded_quad_with_oes_texture, "corner_rad").set_vec4(quad.crad);
    draw_quad(rounded_quad_with_oes_texture, quad.bounds, quad.uv_bounds, quad.uv_rot);
}
#endif

#ifdef DEBUG
void engine_backend::DEBUG_draw_marker(const vec2& pt, const vec3& col) {
    DEBUG_marker_queue.push_back({pt, col});
}

void engine_backend::DEBUG_draw() {
    use_program(DEBUG_marker_program);

    for(const DEBUG_marker& marker: DEBUG_marker_queue) {
        get_variable(DEBUG_marker_program, "color").set_vec3(marker.color);
        draw_quad(DEBUG_marker_program, rect::from_middle_and_size(marker.pos, DEBUG_marker_size));
    }

    DEBUG_marker_queue.clear();
}
#endif

scoped_camera_matrix::scoped_camera_matrix(engine_backend* backend, const mat4& camera_matrix) {
    this->backend = backend;

    previous_matrix = backend->projection_mat;
    backend->projection_mat = camera_matrix;
}

scoped_camera_matrix::~scoped_camera_matrix() {
    backend->projection_mat = previous_matrix;
}

s32 even_to_uneven(s32 N) {
    s32 M;
    while((M = N / 2 + 1) % 2 == 0 || N % 2 == 0 || M < 1) N++;
    return N;
}
    
void texture_sampler::init(engine_backend* backend, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, 
              const vertex* vertices, u32 vertices_size, const u32* indices, u32 indices_size) {
    this->backend = backend;
    this->input_is_oes_texture = input_is_oes_texture;
    this->input_tex = input_tex;
    this->output_size = output_size;
    this->vertices = vertices;
    this->vertices_size = vertices_size;
    this->indices = indices;
    this->indices_size = indices_size;

    projection_matrix = mat4::orthographic(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

    LOGI("sampler output_size: %d, %d", output_size.x, output_size.y);
    output_tex = make_texture(output_size, GL_RGBA32F);    
    output_fb = framebuffer_from_texture(output_tex);

    buffer = make_shader_buffer();
    fill_shader_buffer(buffer, vertices, vertices_size * sizeof(vertex), indices, indices_size * sizeof(u32));

    sampler_program = backend->compile_and_link(vert_src(), frag_sampler_src(input_is_oes_texture));
}

void texture_sampler::sample() {
    SCOPED_CAMERA_MATRIX(backend, projection_matrix);
   
    fill_shader_buffer(buffer, vertices, vertices_size * sizeof(vertex), null, 0);

    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    bind_shader_buffer(buffer);
    bind_framebuffer(output_fb);

    if(!input_is_oes_texture) {
        bind_texture_to_slot(0, *input_tex);
    }

    backend->use_program(sampler_program);

    glViewport(0, 0, output_size.x, output_size.y);
    glDrawElements(GL_TRIANGLES, indices_size, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");

    // restore old viewport
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

void texture_downsampler_stage::init(engine_backend* backend, svec2 input_size, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor) {
    this->backend = backend;
    this->input_size = input_size;
    this->output_size = output_size;
    this->input_is_oes_texture = input_is_oes_texture;
    this->input_tex = input_tex;

    svec2 req_kernel_size = {
        (s32)((f32)input_size.x / (relaxation_factor * (f32)output_size.x)),
        (s32)((f32)input_size.y / (relaxation_factor * (f32)output_size.y))
    };

    req_kernel_size.x = even_to_uneven(req_kernel_size.x);
    req_kernel_size.y = even_to_uneven(req_kernel_size.y);

    temp_tex = make_texture({ output_size.x, input_size.y }, GL_RGBA16F);
    output_tex = make_texture(output_size, GL_RGBA32F);
    
    temp_fb = framebuffer_from_texture(temp_tex);
    output_fb = framebuffer_from_texture(output_tex);
    
    std::string gauss_frag_src_x = frag_gauss_blur_src(input_is_oes_texture, req_kernel_size.x, {1.0f / (f32)input_size.x, 0.0f});
    gauss_blur_x_program = backend->compile_and_link(vert_src(), gauss_frag_src_x);
    ASSERT(gauss_blur_x_program.program, "gauss_blur_x_program program could not be compiled.");
    
    std::string gauss_frag_src_y = frag_gauss_blur_src(false, req_kernel_size.y, {0.0f, 1.0f / (f32)input_size.y});
    gauss_blur_y_program = backend->compile_and_link(vert_src(), gauss_frag_src_y);
    ASSERT(gauss_blur_y_program.program, "gauss_blur_y_program program could not be compiled.");
}

void texture_downsampler_stage::downsample() {
    // first blur pass    
    backend->use_program(gauss_blur_x_program);
    bind_framebuffer(temp_fb);

    if(!input_is_oes_texture) {
        bind_texture_to_slot(0, *input_tex);
    }

    glViewport(0, 0, output_size.x, input_size.y);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");


    // second blur pass
    backend->use_program(gauss_blur_y_program);
    bind_texture_to_slot(0, temp_tex);
    bind_framebuffer(output_fb);
    
    glViewport(0, 0, output_size.x, output_size.y);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");
}

void docscanner::texture_downsampler::init(engine_backend* backend, svec2 input_size, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, s32 downsampling_stages, f32 relaxation_factor) {
    this->backend = backend;
    this->input_size = input_size;
    this->output_size = output_size;
    this->input_is_oes_texture = input_is_oes_texture;
    this->input_tex = input_tex;

    projection_matrix = mat4::orthographic(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

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

    stages = new texture_downsampler_stage[downsampling_stages];
    stages_size = downsampling_stages;


    svec2 downsampling_fac = {
        (s32)sqrt(input_size.x / (f32)output_size.x),
        (s32)sqrt(input_size.y / (f32)output_size.y),
    };

    svec2 stage_input_size = {};
    svec2 stage_output_size = input_size;
    bool stage_input_is_oes_texture = input_is_oes_texture;
    const texture* stage_input_tex = input_tex;
    
    for(s32 i = 0; i < stages_size; i++) {
        texture_downsampler_stage& stage = stages[i]; 

        stage_input_size = stage_output_size;
        stage_output_size = { stage_input_size.x / downsampling_fac.x, stage_input_size.y / downsampling_fac.y };
        if(i == downsampling_stages - 1) stage_output_size = output_size;

        stage.init(backend, stage_input_size, stage_output_size, stage_input_is_oes_texture, stage_input_tex, relaxation_factor);

        stage_input_is_oes_texture = false;
        stage_input_tex = &stage.output_tex;
    }

    const texture_downsampler_stage& last_stage = stages[stages_size - 1];
    output_tex = &last_stage.output_tex;
    output_fb = &last_stage.output_fb;

    fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void docscanner::texture_downsampler::downsample() {
    SCOPED_CAMERA_MATRIX(backend, projection_matrix);
   
    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glBindVertexArray(gauss_quad_buffer.vao);

    for(s32 i = 0; i < stages_size; i++) {
        stages[i].downsample();
    }

    // restore old viewport
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

instanced_shader_buffer make_lines_buffer() {
    u32 indices[6] = { 
        0, 1, 2, 
        0, 2, 3 
    };

    vertex vertices[5] = {
        {{0, -0.5}, {0, 0}},
        {{1, -0.5}, {1, 0}},
        {{1, +0.5}, {1, 1}},
        {{0, +0.5}, {0, 1}}
    };

    shader_buffer quad_buffer = make_shader_buffer();
    fill_shader_buffer(quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    return make_instanced_line_shader_buffer(quad_buffer);
}

instanced_shader_buffer make_joins_buffer(s32 resolution) {
    vertex *vertices = new vertex[resolution];

    for(s32 i = 0; i < resolution; i++) {
        vertex& vertex = vertices[i];
        f32 alpha = (2.0f * M_PI * i) / (f32)resolution;
        f32 x = 0.5f * cosf(alpha);
        f32 y = 0.5f * sinf(alpha);

        vertex.pos = { x, y };
        vertex.uv = { 0.5f, 0.5f };
    }

    shader_buffer round_buffer = make_shader_buffer();
    fill_shader_buffer(round_buffer, vertices, resolution * sizeof(vertex), null, 0);

    delete[] vertices;

    return make_instanced_point_shader_buffer(round_buffer);
}

void docscanner::lines::init(engine_backend* backend, vec2* points, s32 points_size, f32 thickness, vec3 color, bool is_closed) {
    this->backend = backend;

    this->points = points;
    this->is_closed = is_closed;
    
    if(is_closed) {
        this->points_size = points_size + 1;
        closed_points = new vec2[points_size];
    } else {
        this->points_size = points_size;
        closed_points = points;
    }

    this->thickness = thickness;
    this->color = color;

    lines_buffer = make_lines_buffer();
    joins_buffer = make_joins_buffer(16);

    lines_program = backend->compile_and_link(vert_instanced_line_src(), frag_border_src());
    joins_program = backend->compile_and_link(vert_instanced_point_src(), frag_border_src());

    fill();
}

void docscanner::lines::fill() {
    glBindBuffer(GL_ARRAY_BUFFER, joins_buffer.instance_vbo);
    glBufferData(GL_ARRAY_BUFFER, (points_size - 1) * sizeof(vec2), points, GL_DYNAMIC_DRAW);

    if(is_closed) {
        memcpy(closed_points, points, (points_size - 1) * sizeof(vec2));
        closed_points[points_size - 1] = closed_points[0];
    }

    glBindBuffer(GL_ARRAY_BUFFER, lines_buffer.instance_vbo);
    glBufferData(GL_ARRAY_BUFFER, points_size * sizeof(vec2), closed_points, GL_DYNAMIC_DRAW);
}

void docscanner::lines::draw() {
    backend->use_program(joins_program);
    get_variable(joins_program, "scale").set_f32(thickness);
    get_variable(joins_program, "color").set_vec4(color);
    
    glBindVertexArray(joins_buffer.vao);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 16, points_size - 1);

    backend->use_program(lines_program);
    get_variable(lines_program, "thickness").set_f32(thickness);
    get_variable(joins_program, "color").set_vec4(color);
    
    glBindVertexArray(lines_buffer.vao);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null, points_size - 1);
}


bool link_program(GLuint &program) {
    glLinkProgram(program);
    GLint linkStatus = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);

    if (linkStatus == GL_TRUE) return true;
    else {
        GLint bufLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
        if (bufLength) {
            char* buf = (char*) malloc((size_t) bufLength);
            if (buf) {
                glGetProgramInfoLog(program, bufLength, nullptr, buf);
                LOGE("Could not link preview_program:\n%s\n", buf);
                free(buf);
            }
        }

        glDeleteProgram(program);
        program = 0;
        return false;
    }
}

u32 docscanner::compile_shader(u32 type, const char* src) {
    u32 shader = glCreateShader(type);

    if (shader) {
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

        if (!compiled) {
            GLint infoLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);

            if (infoLen) {
                char* buf = (char*) malloc((size_t) infoLen);
                if (buf) {
                    glGetShaderInfoLog(shader, infoLen, nullptr, buf);
                    LOGE("Could not compile shader %d:\n%s\n", type, buf);
                    free(buf);
                }
                glDeleteShader(shader);
                shader = 0;
            }
        }
    }

    ASSERT(shader, "Compile shader failed with source: %s", src);

    return shader;
}

shader_program docscanner::compile_and_link_program(u32 vert_shader, u32 frag_shader) {
    GLuint program = glCreateProgram();
    ASSERT(program, "Compile program failed.");

    glAttachShader(program, vert_shader);
    check_gl_error("glAttachShader");
    glAttachShader(program, frag_shader);
    check_gl_error("glAttachShader");
    
    link_program(program);
    glDetachShader(program, vert_shader);
    check_gl_error("glDetachShader");
    glDetachShader(program, frag_shader);
    check_gl_error("glDetachShader");
    
    return {program};
}

shader_program docscanner::compile_and_link_program(u32 comp_shader) {
    GLuint program = glCreateProgram();
    ASSERT(program, "Compile program failed.");

    glAttachShader(program, comp_shader);
    check_gl_error("glAttachShader");
    
    link_program(program);
    
    glDetachShader(program, comp_shader);
    check_gl_error("glDetachShader");
    
    return {program};
}

bool docscanner::load_program_from_binary(file_context* ctx, u64 hash, shader_program& program) {
    return false;
    std::string path = "shader_binary_" + std::to_string(hash);

    u8* data = null;
    u32 size = 0;
    read_from_internal_file(ctx, path.c_str(), data, size);
    if(!data) return false;

    GLenum bin_format = *((GLenum*)data);
    data += sizeof(GLenum);
    size -= sizeof(GLenum);

    program = { glCreateProgram() }; 
    check_gl_error("glCreateProgram");
    glProgramBinary(program.program, bin_format, data, size); // todo: add binary format
    check_gl_error("glProgramBinary");

    delete[] data;
    
    GLint success;
    glGetProgramiv(program.program, GL_LINK_STATUS, &success);
    check_gl_error("glGetProgramiv");

    return true;
}

void docscanner::save_program_to_binary(file_context* ctx, u64 hash, shader_program& program) {
    return;

    GLint bin_size = -1;
    glGetProgramiv(program.program, GL_PROGRAM_BINARY_LENGTH, &bin_size);
    bin_size += sizeof(GLenum);

    u8* buffer = new u8[bin_size];
    GLsizei bytes_written;
    GLenum bin_format = -1;
    glGetProgramBinary(program.program, bin_size, &bytes_written, &bin_format, buffer + sizeof(GLenum));
    *((GLenum*)buffer) = bin_format;
    
    std::string path = "shader_binary_" + std::to_string(hash);
    write_to_internal_file(ctx, path.c_str(), buffer, bin_size);

    delete[] buffer;
}

void delete_shader(u32 id) {
    if(id) {
        glDeleteShader(id);
    }
}

void docscanner::delete_program(shader_program &program) {
    if (program.program) {
        glUseProgram(0);
        glDeleteProgram(program.program);
        program.program = 0;
    }
}

void docscanner::dispatch_compute_program(const svec2 size, u32 depth) {
    glDispatchCompute((u32)size.x, (u32)size.y, depth);
    check_gl_error("glDispatchCompute");

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    check_gl_error("glFinish");
}

shader_buffer docscanner::make_shader_buffer() {
    shader_buffer buffer = {};

    glGenVertexArrays(1, &buffer.vao);
    glBindVertexArray(buffer.vao);

    GLuint buffers[2];  
    glGenBuffers(2, buffers);

    buffer.vbo = buffers[0];
    glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);

#define attrib_enable(index, num_comp, var_name) \
        glVertexAttribPointer(index, num_comp, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*) offsetof(vertex, var_name)); \
        glEnableVertexAttribArray(index)

    attrib_enable(0, 2, pos);
    attrib_enable(1, 2, uv);

#undef attrib_enable

    buffer.ebo = buffers[1];
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.ebo);

    return buffer;
}

instanced_shader_buffer docscanner::make_instanced_quad_shader_buffer(shader_buffer buff) {
    bind_shader_buffer(buff);

    GLuint instance_vbo;
    glGenBuffers(1, &instance_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);

#define attrib_enable(index, num_comp, var_name) \
        glVertexAttribPointer(vertex_attrib_count + index, num_comp, GL_FLOAT, GL_FALSE, sizeof(instanced_quad), (void*) offsetof(instanced_quad, var_name)); \
        glEnableVertexAttribArray(vertex_attrib_count + index); \
        glVertexAttribDivisor(vertex_attrib_count + index, 1)

    attrib_enable(0, 2, v0);
    attrib_enable(1, 2, v1);
    attrib_enable(2, 2, v2);
    attrib_enable(3, 2, v3);

    attrib_enable(4, 2, uv_tl);
    attrib_enable(5, 2, uv_br);

#undef attrib_enable

    return {
        .vao=buff.vao,
        .instance_vbo=instance_vbo
    };
}

GLuint prep_instance_vbo(const shader_buffer& buff) {
    bind_shader_buffer(buff);

    GLuint instance_vbo;
    glGenBuffers(1, &instance_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);

    return instance_vbo;
}

instanced_shader_buffer docscanner::make_instanced_point_shader_buffer(shader_buffer buff) {
    u32 instance_vbo = prep_instance_vbo(buff);

#define attrib_enable(index, num_comp, offset) \
        glVertexAttribPointer(vertex_attrib_count + index, num_comp, GL_FLOAT, GL_FALSE, sizeof(vec2), (void*) offset); \
        glEnableVertexAttribArray(vertex_attrib_count + index); \
        glVertexAttribDivisor(vertex_attrib_count + index, 1)

    attrib_enable(0, 2, 0);

#undef attrib_enable

    return {
        .vao=buff.vao,
        .instance_vbo=instance_vbo
    };
}

instanced_shader_buffer docscanner::make_instanced_line_shader_buffer(shader_buffer buff) {
    u32 instance_vbo = prep_instance_vbo(buff);

#define attrib_enable(index, num_comp, offset) \
        glVertexAttribPointer(vertex_attrib_count + index, num_comp, GL_FLOAT, GL_FALSE, sizeof(vec2), (void*) offset); \
        glEnableVertexAttribArray(vertex_attrib_count + index); \
        glVertexAttribDivisor(vertex_attrib_count + index, 1)

    attrib_enable(0, 2, 0);
    attrib_enable(1, 2, sizeof(vec2));

#undef attrib_enable

    return {
        .vao=buff.vao,
        .instance_vbo=instance_vbo
    };
}

void docscanner::fill_shader_buffer(const shader_buffer& buff, const vertex* vertices, u32 vertices_size, const u32* indices, u32 indices_size) {
    ASSERT(vertices || indices, "Neither vertices nor indices can be updated.");

    if(vertices) {
        glBindBuffer(GL_ARRAY_BUFFER, buff.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        check_gl_error("glBufferData");
    }
        
    if(indices) {
        glBindVertexArray(buff.vao);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);
        check_gl_error("glBufferData");
    }
}

void docscanner::bind_shader_buffer(const shader_buffer& buff) {
    glBindVertexArray(buff.vao);
    check_gl_error("glBindVertexArray");
}

texture docscanner::make_texture(svec2 size, u32 format) {
    u32 id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexStorage2D(GL_TEXTURE_2D, 1, format, (int) size.x, (int) size.y);
    check_gl_error("glTexStorage2D");
    return {id, format, size};
}

stack_texture docscanner::make_stack_texture(s32 depth, svec2 size, u32 format) {
    u32 id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D_ARRAY, id);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, format, (int) size.x, (int) size.y, depth);
    check_gl_error("glTexStorage3D");

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    check_gl_error("glTexParameteri");
    
    return {id, format, size, depth};
}

void docscanner::bind_image_to_slot(u32 slot, const texture &tex) {
    glBindImageTexture(slot, tex.id, 0, GL_FALSE, 0, GL_WRITE_ONLY, tex.format);
    check_gl_error("glBindImageTexture");
}

void docscanner::bind_texture_to_slot(u32 slot, const texture &tex) {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, tex.id);
    check_gl_error("glBindTexture");
}

void docscanner::bind_texture_to_slot(u32 slot, const stack_texture &tex) {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex.id);
    check_gl_error("glBindTexture");
}

void docscanner::bind_framebuffer(const frame_buffer& fb) {
    glBindFramebuffer(GL_FRAMEBUFFER, fb.id);
    check_gl_error("glBindFramebuffer");
}

void docscanner::unbind_framebuffer() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    check_gl_error("glBindFramebuffer");
}

frame_buffer docscanner::framebuffer_from_texture(const texture& tex) {
    GLuint fb;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.id, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    ASSERT(status == GL_FRAMEBUFFER_COMPLETE, "glCheckFramebufferStatus indicates the Framebuffer is incomplete (error code 0x%x).", status);

    return {fb};
}

void docscanner::get_framebuffer_data(const frame_buffer &fb, const svec2 &size, u8* &data, u32 data_size) { 
    bind_framebuffer(fb);
    glReadPixels(0, 0, size.x, size.y, GL_RGBA, GL_FLOAT, data);
    check_gl_error("glReadPixels");
}

void get_texture_format_and_type(u32 tex_format, GLenum& format, GLenum& type) {
    switch(tex_format) {
    case GL_R32F: {
        format = GL_RED;
        type = GL_FLOAT;
    } break;
    case GL_RGBA32F: { 
        format = GL_RGBA;
        type = GL_FLOAT;
    } break;
    default: LOGE_AND_BREAK("Unsupported texture format in set_texture_data.");
    }
}

void docscanner::set_texture_data(const texture &tex, u8* data, const svec2& size) {
    GLenum format = 0, type = 0;
    get_texture_format_and_type(tex.format, format, type);

    glBindTexture(GL_TEXTURE_2D, tex.id);
    check_gl_error("glBindTexture");

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.x, size.y, format, type, data);
    check_gl_error("glTexSubImage2D");
}

void docscanner::set_texture_data(const stack_texture &tex, u8* data, const svec2& size) {
    GLenum format = 0, type = 0;
    get_texture_format_and_type(tex.format, format, type);

    glBindTexture(GL_TEXTURE_2D_ARRAY, tex.id);
    check_gl_error("glBindTexture");

    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, size.x, size.y, tex.depth, format, type, data);
    check_gl_error("glTexSubImage3D");
}

variable docscanner::get_variable(const shader_program& program, const char* name) {
    int location = glGetUniformLocation(program.program, name);
    check_gl_error("glGetUniformLocation");

    return { location };
}

#endif