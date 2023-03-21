#if 1
#include "log.hpp"
#include "backend.hpp"
// #include <cstdlib> // todo: uncomment this
// #include <csignal>
// #include <string>

#if defined(LINUX)
#include <cstddef>
#endif

using namespace docscanner;

void docscanner::check_gl_error(const char* op) {
    for (GLenum error = glGetError(); error; error = glGetError()) {
        LOGE_AND_BREAK("glError with code 0x%04x was triggered by %s().\n", error, op);
    }
}

void docscanner::variable::set_f32(f32 v) {
    glUniform1f(location, v);
}

void docscanner::variable::set_mat4(float* data) {
    glUniformMatrix4fv(location, 1, GL_FALSE, data);
}

void docscanner::variable::set_vec2(const vec2& v) {
    glUniform2f(location, v.x, v.y);
}

void docscanner::texture_downsampler::init(shader_programmer* programmer, uvec2 input_size, uvec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor) {
    this->programmer = programmer;
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

    temp_tex = create_texture({output_size.x, input_size.y}, GL_RGBA16F);
    output_tex = create_texture(output_size, GL_RGBA32F);
    
    temp_fb = framebuffer_from_texture(temp_tex);
    output_fb = framebuffer_from_texture(output_tex);
    
    std::string gauss_frag_src_x = frag_gauss_blur_src(input_is_oes_texture, req_kernel_size.x, {1.0f / (f32)input_size.x, 0.0f});
    gauss_blur_x_program = programmer->compile_and_link(vert_src, gauss_frag_src_x);
    ASSERT(gauss_blur_x_program.program, "gauss_blur_x_program program could not be compiled.");
    
    std::string gauss_frag_src_y = frag_gauss_blur_src(false, req_kernel_size.y, {0.0f, 1.0f / (f32)input_size.y});
    gauss_blur_y_program = programmer->compile_and_link(vert_src, gauss_frag_src_y);
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
            const vec2& tl = (*stick_vertices)[x * stick_size.y + y].pos;
            const vec2& tr = (*stick_vertices)[(x + 1) * stick_size.y + y].pos;
            const vec2& br = (*stick_vertices)[(x + 1) * stick_size.y + (y + 1)].pos;
            const vec2& bl = (*stick_vertices)[x * stick_size.y + (y + 1)].pos;

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

void sticky_particle_system::init(shader_programmer* programmer, const std::vector<vertex>& stick_vertices, const svec2& stick_size, float* projection, shader_buffer buffer) {
    this->stick_vertices = &stick_vertices;
    this->stick_size = stick_size;
    this->buffer = buffer;

    shader = programmer->compile_and_link(vert_src, frag_particle_src);

    use_program(shader);
    get_variable(shader, "projection").set_mat4(projection);
}

void sticky_particle_system::render() {
    if(stick_vertices->size() != stick_size.area()) return;

    use_program(shader);

    gen_and_fill_mesh_vertices();
    glBindVertexArray(buffer.id);

    glDrawElements(GL_TRIANGLES, mesh_indices.size(), GL_UNSIGNED_INT, null);
}

constexpr f32 thickness = 0.1f;
constexpr f32 half_thickness = thickness / 2.0f;

void push_border_vertices(std::vector<vertex>& mesh_vertices, const std::vector<vertex>& border_vertices,
                          s32 start_index, s32 stride, s32 end_index, s32 i) {
    const vec2& curr = border_vertices[i].pos;
    const vec2& last = (i - stride < start_index) ? curr : border_vertices[i - stride].pos;
    const vec2& next = (i + stride > end_index) ? curr : border_vertices[i + stride].pos;
            
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

void push_border_vertices_forward(std::vector<vertex>& mesh_vertices, const std::vector<vertex>& border_vertices,
                                  s32 start_index, s32 stride, s32 end_index) {
    for(s32 i = start_index; i <= end_index; i += stride) {
        push_border_vertices(mesh_vertices, border_vertices, start_index, stride, end_index, i);
    }
}

void push_border_vertices_backward(std::vector<vertex>& mesh_vertices, const std::vector<vertex>& border_vertices,
                                   s32 start_index, s32 stride, s32 end_index) {
    for(s32 i = end_index; i >= start_index; i -= stride) {
        push_border_vertices(mesh_vertices, border_vertices, start_index, stride, end_index, i);
    }
}

void mesh_border::gen_and_fill_mesh_vertices() {
    mesh_indices.clear();
    mesh_vertices.clear();

    for(s32 i = 0; i < 2 * border_size.x + 2 * border_size.y - 1; i++) {
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

    push_border_vertices_forward(mesh_vertices, *border_vertices, 
        0, 
        border_size.y, 
        (border_size.x - 1) * border_size.y
    );

    push_border_vertices_forward(mesh_vertices, *border_vertices, 
        (border_size.x - 1) * border_size.y, 
        1, 
        (border_size.y - 1) + (border_size.x - 1) * border_size.y
    );
    
    push_border_vertices_backward(mesh_vertices, *border_vertices, 
        (border_size.y - 1), 
        border_size.y, 
        (border_size.y - 1) + (border_size.x - 1) * border_size.y
    );

    push_border_vertices_backward(mesh_vertices, *border_vertices, 
        0, 
        1, 
        (border_size.y - 1)
    );

    fill_shader_buffer(buffer, mesh_vertices.data(), mesh_vertices.size() * sizeof(vertex), mesh_indices.data(), mesh_indices.size() * sizeof(u32));
}

void mesh_border::init(shader_programmer* programmer, const std::vector<vertex>& border_vertices, const svec2& border_size, float* projection, shader_buffer buffer) {
    this->border_vertices = &border_vertices;
    this->border_size = border_size;
    this->buffer = buffer;

    shader = programmer->compile_and_link(vert_src, frag_border_src);

    use_program(shader);
    get_variable(shader, "projection").set_mat4(projection);

    time_var = get_variable(shader, "time");
}

void mesh_border::render(f32 time) {
    if(border_vertices->size() != border_size.area()) return;

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
    check_gl_error("glDetachSahder");
    
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

void docscanner::fill_shader_buffer(const shader_buffer& buff, vertex* vertices, u32 vertices_size, u32* indices, u32 indices_size) {
    glBindVertexArray(buff.id);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);
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

frame_buffer docscanner::framebuffer_from_texture(const texture& tex) {
    GLuint fb;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.id, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    ASSERT(status == GL_FRAMEBUFFER_COMPLETE, "glCheckFramebufferStatus indicates the Framebuffer is incomplete (error code 0x%x).", status);

    return {fb};
}

void docscanner::get_framebuffer_data(const frame_buffer &fb, const uvec2 &size, u8* &data, u32 data_size) { 
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