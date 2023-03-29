#if 1
#include "log.hpp"
#include "backend.hpp"

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

void docscanner::variable::set_vec4(const vec2& a, const vec2& b) {
    glUniform4f(location, a.x, a.y, b.x, b.y);
}

docscanner::instanced_quads::instanced_quads(s32 size) {
    quads = new instanced_quad[size];
    quads_size = size;
    
    u32 indices[6] = { 
        0, 1, 2, 
        0, 2, 3 
    };

    vec2 pos = {0.5f, 0.5f};
    vertex vertices[5] = {
        {{0, 0}, {0, 0}},
        {{1, 0}, {1, 0}},
        {{1, 1}, {1, 1}},
        {{0, 1}, {0, 1}}
    };

    shader_buffer quad_buffer = make_shader_buffer();
    fill_shader_buffer(quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    quads_buffer = make_instanced_quad_shader_buffer(quad_buffer);

    glBindBuffer(GL_ARRAY_BUFFER, quads_buffer.instance_vbo);
    glBufferData(GL_ARRAY_BUFFER, size * sizeof(instanced_quad), quads, GL_DYNAMIC_DRAW);
}

void docscanner::instanced_quads::draw() {
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null, quads_size);
}

void docscanner::engine_backend::init(mat4 projection_mat) {
    this->projection_mat = projection_mat;

    glEnable(GL_BLEND);  
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    u32 indices[6] = { 
        0, 1, 2, 
        0, 2, 3 
    };

    vec2 pos = {0.5f, 0.5f};
    vertex vertices[5] = {
        {{0, 0}, {0, 0}},
        {{1, 0}, {1, 0}},
        {{1, 1}, {1, 1}},
        {{0, 1}, {0, 1}}
    };

    quad_buffer = make_shader_buffer();
    fill_shader_buffer(quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

#ifdef DEBUG
    DEBUG_marker_program = compile_and_link(vert_quad_src, frag_DEBUG_marker_src);
#endif
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

shader_program docscanner::engine_backend::compile_and_link(const std::string& vert_src, const std::string& frag_src) {
    u32 vert_shader = find_or_insert_shader(shader_map, GL_VERTEX_SHADER, vert_src);
    u32 frag_shader = find_or_insert_shader(shader_map, GL_FRAGMENT_SHADER, frag_src);

    u64 program_id = ((u64)vert_shader << 32) | (u64)frag_shader;
    auto program_found = program_map.find(program_id);
    if(true || program_found == program_map.end()) {
        shader_program program = compile_and_link_program(vert_shader, frag_shader);
        program_map.emplace(program_id, program.program);

        use_program(program);
        get_variable(program, "projection").set_mat4(projection_mat);

        return program;
    } else {
        return {program_found->second};
    }
}


shader_program docscanner::engine_backend::compile_and_link(const std::string& comp_src) {
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

void docscanner::engine_backend::draw_quad(const vec2& pos, const vec2& size) {
    bind_shader_buffer(quad_buffer);

    get_variable(DEBUG_marker_program, "transform").set_vec4(pos, size);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");
}

void draw_instanced_quads(instanced_quad* quads, u32 quads_size) {

}

#ifdef DEBUG
void docscanner::engine_backend::DEBUG_draw_marker(const vec2& pt, const vec3& col) {
    DEBUG_marker_queue.push_back({pt, col});
}

void docscanner::engine_backend::DEBUG_draw() {
    LOGI("DEBUG_draw %u markers.", (u32)DEBUG_marker_queue.size());
    use_program(DEBUG_marker_program);

    for(const DEBUG_marker& marker: DEBUG_marker_queue) {
        get_variable(DEBUG_marker_program, "color").set_vec3(marker.color);
        draw_quad(marker.pos - DEBUG_marker_size * 0.5, DEBUG_marker_size);
    }

    DEBUG_marker_queue.clear();
}
#endif

void docscanner::texture_downsampler::init(engine_backend* backend, uvec2 input_size, svec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->input_is_oes_texture = input_is_oes_texture;
    this->input_tex = input_tex;

#define EVEN_TO_UNEVEN(n) if ((n) % 2 == 0) { n++; }

    uvec2 req_kernel_size = {
        (u32)((f32)input_size.x / (relaxation_factor * (f32)output_size.x)),
        (u32)((f32)input_size.y / (relaxation_factor * (f32)output_size.y))
    };

    EVEN_TO_UNEVEN(req_kernel_size.x);
    EVEN_TO_UNEVEN(req_kernel_size.y);

#undef EVEN_TO_UNEVEN

    temp_tex = create_texture({(u32)output_size.x, (u32)input_size.y}, GL_RGBA16F);
    output_tex = create_texture({(u32)output_size.x, (u32)output_size.y}, GL_RGBA32F);
    
    temp_fb = framebuffer_from_texture(temp_tex);
    output_fb = framebuffer_from_texture(output_tex);
    
    std::string gauss_frag_src_x = frag_gauss_blur_src(input_is_oes_texture, req_kernel_size.x, {1.0f / (f32)input_size.x, 0.0f});
    gauss_blur_x_program = backend->compile_and_link(vert_src, gauss_frag_src_x);
    ASSERT(gauss_blur_x_program.program, "gauss_blur_x_program program could not be compiled.");
    
    std::string gauss_frag_src_y = frag_gauss_blur_src(false, req_kernel_size.y, {0.0f, 1.0f / (f32)input_size.y});
    gauss_blur_y_program = backend->compile_and_link(vert_src, gauss_frag_src_y);
    ASSERT(gauss_blur_y_program.program, "gauss_blur_y_program program could not be compiled.");

    mat4 projection_mat;
    mat4f_load_ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f, projection_mat.data);

    use_program(gauss_blur_x_program);
    auto proj_matrix_x_var = get_variable(gauss_blur_x_program, "projection");
    proj_matrix_x_var.set_mat4(projection_mat);

    use_program(gauss_blur_y_program);
    auto proj_matrix_y_var = get_variable(gauss_blur_y_program, "projection");    
    proj_matrix_y_var.set_mat4(projection_mat);

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

texture* docscanner::texture_downsampler::downsample() {
    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glBindVertexArray(gauss_quad_buffer.id);

    // first blur pass    
    use_program(gauss_blur_x_program);
    bind_framebuffer(temp_fb);

    if(!input_is_oes_texture) {
        bind_texture_to_slot(0, *input_tex);
    }

    glViewport(0, 0, output_size.x, input_size.y);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");

    // second blur pass
    use_program(gauss_blur_y_program);
    bind_texture_to_slot(0, temp_tex);
    bind_framebuffer(output_fb);
    
    glViewport(0, 0, output_size.x, output_size.y);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
    check_gl_error("glDrawElements");

    // restore old viewport
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

    return &output_tex;
}

void sticky_particle_system::gen_and_fill_mesh_vertices() {
    mesh_indices.clear();
    mesh_vertices.clear();

    for(s32 i = 0; i < (stick_size.x - 1) * (stick_size.y - 1); i++) {
        s32 offset = i * 4;
        mesh_indices.push_back(0 + offset);
        mesh_indices.push_back(1 + offset);
        mesh_indices.push_back(2 + offset);

        mesh_indices.push_back(0 + offset);
        mesh_indices.push_back(2 + offset);
        mesh_indices.push_back(3 + offset);
    }

    for(s32 x = 0; x < stick_size.x - 1; x++) {
        for(s32 y = 0; y < stick_size.y - 1; y++) {
            const vec2& tl = stick_vertices[x * stick_size.y + y].pos;
            const vec2& tr = stick_vertices[(x + 1) * stick_size.y + y].pos;
            const vec2& br = stick_vertices[(x + 1) * stick_size.y + (y + 1)].pos;
            const vec2& bl = stick_vertices[x * stick_size.y + (y + 1)].pos;

            const vec2 middle = (tl + tr + br + bl) * (1.0f / 4.0f);

#define particle_scale 0.3
#define rescale_particle(pt) ((pt - middle) * particle_scale + middle)

            mesh_vertices.push_back({rescale_particle(tl), {0, 0}});
            mesh_vertices.push_back({rescale_particle(tr), {1, 0}});
            mesh_vertices.push_back({rescale_particle(br), {1, 1}});
            mesh_vertices.push_back({rescale_particle(bl), {0, 1}});
        
#undef rescale_particle
        }
    }

    fill_shader_buffer(buffer, mesh_vertices.data(), mesh_vertices.size() * sizeof(vertex), mesh_indices.data(), mesh_indices.size() * sizeof(u32));
}

void sticky_particle_system::init(engine_backend* backend, const vertex* vertices, const svec2& stick_size, shader_buffer buffer) {
    this->stick_vertices = vertices;
    this->stick_size = stick_size;
    this->buffer = buffer;

    shader = backend->compile_and_link(vert_src, frag_particle_src);
}

void sticky_particle_system::render() {
    use_program(shader);

    gen_and_fill_mesh_vertices();
    glBindVertexArray(buffer.id);

    glDrawElements(GL_TRIANGLES, mesh_indices.size(), GL_UNSIGNED_INT, null);
}

constexpr f32 thickness = 0.1f;
constexpr f32 half_thickness = thickness / 2.0f;

void push_border_vertices(std::vector<vertex>& mesh_vertices, const vec2* border_pts, s32 border_size, s32 i) {
    const vec2& curr = border_pts[i];
    const vec2& last = (i - 1 < 0) ? curr : border_pts[i - 1];
    const vec2& next = (i + 1 > border_size) ? curr : border_pts[i + 1];
            
    const vec2 normal = ((curr - last) * 0.5 + (next - curr) * 0.5).orthogonal().normalize();
    const vec2 half_border = normal * half_thickness;

    const vec2 curr_small = curr - half_border;
    const vec2 curr_large = curr + half_border;

    vec2 ref = { 1.0f, 0.0f };
    f32 curr_angle = vec2::angle_between(curr, ref);

    mesh_vertices.push_back({curr, {0.5f, curr_angle}});
    mesh_vertices.push_back({curr_small, {0.0f, curr_angle}});
    mesh_vertices.push_back({curr_large, {1.0f, curr_angle}});
}

void push_border_vertices_forward(std::vector<vertex>& mesh_vertices, const vec2* border_pts, s32 border_size) {
    for(s32 i = 0; i < border_size; i++) {
        push_border_vertices(mesh_vertices, border_pts, border_size, i);
    }
}

void push_border_vertices_backward(std::vector<vertex>& mesh_vertices, const vec2* border_pts, s32 border_size) {
    for(s32 i = border_size - 1; i >= 0; i--) {
        push_border_vertices(mesh_vertices, border_pts, border_size, i);
    }
}

void mesh_border::gen_and_fill_mesh_vertices() {
    mesh_indices.clear();
    mesh_vertices.clear();

    for(s32 i = 0; i < 4 * border_size - 1; i++) {
        s32 offset_1 = i * 3;
        s32 offset_2 = i * 3 + 3;

        // inner quad
        mesh_indices.push_back(0 + offset_1);
        mesh_indices.push_back(1 + offset_1);
        mesh_indices.push_back(3 + offset_1);

        mesh_indices.push_back(1 + offset_1);
        mesh_indices.push_back(3 + offset_1);
        mesh_indices.push_back(1 + offset_2);
    
        // outer quad
        mesh_indices.push_back(0 + offset_1);
        mesh_indices.push_back(2 + offset_1);
        mesh_indices.push_back(0 + offset_2);

        mesh_indices.push_back(2 + offset_1);
        mesh_indices.push_back(0 + offset_2);
        mesh_indices.push_back(2 + offset_2);
    }

    // to make it seem complete

    push_border_vertices_forward(mesh_vertices, left_border, border_size);
    push_border_vertices_forward(mesh_vertices, top_border, border_size);
    push_border_vertices_forward(mesh_vertices, right_border, border_size);
    push_border_vertices_forward(mesh_vertices, bottom_border, border_size);
    
    #if false
    push_border_vertices_backward(mesh_vertices, border_vertices, 
        (border_size.y - 1), 
        border_size.y, 
        (border_size.y - 1) + (border_size.x - 1) * border_size.y
    );

    push_border_vertices_backward(mesh_vertices, border_vertices, 
        0, 
        1, 
        (border_size.y - 1)
    );
    #endif

    fill_shader_buffer(buffer, mesh_vertices.data(), mesh_vertices.size() * sizeof(vertex), mesh_indices.data(), mesh_indices.size() * sizeof(u32));
}

void docscanner::mesh_border::init(engine_backend* backend, const vec2* left_border, const vec2* top_border, const vec2* right_border, const vec2* bottom_border, s32 border_size, shader_buffer buffer) {
    this->left_border = left_border;
    this->top_border = top_border;
    this->right_border = right_border;
    this->bottom_border = bottom_border;
    this->border_size = border_size;    
    this->buffer = buffer;

    shader = backend->compile_and_link(vert_src, frag_border_src);

    time_var = get_variable(shader, "time");
}

void mesh_border::render(f32 time) {
    use_program(shader);

    gen_and_fill_mesh_vertices();
    glBindVertexArray(buffer.id);

    time_var.set_f32(time);

    glDrawElements(GL_TRIANGLES, mesh_indices.size(), GL_UNSIGNED_INT, null);
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
    check_gl_error("glDetachSahder");
    glDetachShader(program, frag_shader);
    check_gl_error("glDetachSahder");
    
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

void docscanner::use_program(const shader_program &program) {
    glUseProgram(program.program);
    check_gl_error("glUseProgram");
}

void docscanner::dispatch_compute_program(const uvec2 size, u32 depth) {
    glDispatchCompute(size.x, size.y, depth);
    check_gl_error("glDispatchCompute");

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    check_gl_error("glFinish");
}

shader_buffer docscanner::make_shader_buffer() {
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint buffers[2];
    glGenBuffers(2, buffers);

    GLuint vbo = buffers[0];
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

#define attrib_enable(index, num_comp, var_name) \
        glVertexAttribPointer(index, num_comp, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*) offsetof(vertex, var_name)); \
        glEnableVertexAttribArray(index)

    attrib_enable(0, 2, pos);
    attrib_enable(1, 2, uv);

#undef attrib_enable

    GLuint ebo = buffers[1];
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

    return {vao};
}

instanced_shader_buffer docscanner::make_instanced_quad_shader_buffer(shader_buffer buff) {
    bind_shader_buffer(buff);

    GLuint instance_vbo;
    glGenBuffers(1, &instance_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);

#define attrib_enable(index, num_comp, var_name) \
        glVertexAttribPointer(vertex_attrib_count + index, num_comp, GL_FLOAT, GL_FALSE, sizeof(instanced_quad), (void*) offsetof(instanced_quad, var_name)); \
        glEnableVertexAttribArray(index); \
        glVertexAttribDivisor(index, 1)

    attrib_enable(0, 2, v0);
    attrib_enable(1, 2, v1);
    attrib_enable(2, 2, v2);
    attrib_enable(3, 2, v3);

#undef attrib_enable

    return {buff.id, instance_vbo};
}

void docscanner::fill_shader_buffer(const shader_buffer& buff, vertex* vertices, u32 vertices_size, u32* indices, u32 indices_size) {
    ASSERT(vertices || indices, "Neither vertices nor indices can be updated.");

    glBindVertexArray(buff.id);

    if(vertices) {
        glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, GL_STATIC_DRAW);
    }
        
    if(indices) {
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);
    }
}

void docscanner::bind_shader_buffer(const shader_buffer& buff) {
    glBindVertexArray(buff.id);
}

texture docscanner::create_texture(uvec2 size, u32 format) {
    u32 id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexStorage2D(GL_TEXTURE_2D, 1, format, (int) size.x, (int) size.y);
    check_gl_error("glTexStorage2D");
    return {id, format};
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

void docscanner::set_texture_data(const texture &tex, u8* data, int width, int height) {
    glBindTexture(GL_TEXTURE_2D, tex.id);

    GLenum format, type;
    switch(tex.format) {
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

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, data);
    check_gl_error("glTexSubImage2D");
}

variable docscanner::get_variable(const shader_program& program, const char* name) {
    int location = glGetUniformLocation(program.program, name);
    check_gl_error("glGetUniformLocation");

    return { location };
}

void docscanner::draw(const canvas &canvas) {
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(canvas.bg_color.x, canvas.bg_color.y, canvas.bg_color.z, 1);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
}

#endif