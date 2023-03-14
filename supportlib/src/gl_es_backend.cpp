#ifdef ANDROID
#include "log.h"
#include "backend.h"
#include <cstdlib>
#include <csignal>
#include <string>
#include <math.h>

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

constexpr const char gauss_blur_frag_src[] = R"(#version 310 es
#if %d
        #extension GL_OES_EGL_image_external_essl3 : require
        uniform layout(binding = 0) samplerExternalOES sampler;
#else
        uniform layout(binding = 0) sampler2D sampler;
#endif

        precision mediump float;
        
        in vec2 out_uvs;
        out vec4 out_col;

        const int M = %u;
        const float coeffs[M] = float[M](%s);
        const vec2 pixel_shift = vec2(%f, %f);

        void main() {
            vec4 col = coeffs[0] * texture(sampler, out_uvs);

            for(int i = 1; i < M; i += 2) {
                float w0 = coeffs[i];
                float w1 = coeffs[i + 1];

                float w = w0 + w1;
                float t = w1 / w;

                col += w * texture(sampler, out_uvs + (float(i) + t) * pixel_shift);
                col += w * texture(sampler, out_uvs - (float(i) + t) * pixel_shift);
            }

            out_col = col;
        }
)";

void docscanner::check_gl_error(const char* op) {
    for (GLenum error = glGetError(); error; error = glGetError()) {
        LOGE_AND_BREAK("glError with code 0x%04x was triggered by %s().\n", error, op);
    }
}

void docscanner::variable::set_mat4(float* data) {
    glUniformMatrix4fv(location, 1, GL_FALSE, data);
}

std::string compute_gauss_coefficients(int n) {
    int m = n / 2 + 1;
    f32 sigma = (n - 1.0f) / 4.0f;
    
    f32 *coeffs = new f32[m];
    f32 sum = 0.0f;

#define compute_gauss_coeff(i) \
        const f32 gauss_factor = 1.0 / sqrt(2 * M_PI * sigma); \
        const f32 x = (f32)(i); \
        coeffs[i] = gauss_factor * pow(M_E, -(x * x) / (2 * sigma * sigma));

    compute_gauss_coeff(0);        

    for(int i = 1; i < m; i++) {
        compute_gauss_coeff(i);
        sum += 2 * coeffs[i];
    } 

#undef compute_gauss_coeff

    // normalize
    for(int i = 0; i < m; i++) {
        coeffs[i] /= sum;
    }

    std::string ret = "";
    for(int i = 0; i < m; i++) {
        ret = ret + std::to_string(coeffs[i]);

        if(i != m - 1) ret = ret + ", ";
    }

    delete[] coeffs;

    return ret;
}

char* prepare_gauss_fragment_src(bool sample_from_external, u32 n, vec2 pixel_shift) {
    std::string gauss_coeffs = compute_gauss_coefficients(n);

    size_t needed = snprintf(null, 0, gauss_blur_frag_src, sample_from_external, n / 2 + 1, gauss_coeffs.c_str(), pixel_shift.x, pixel_shift.y);
    
    char* buff = new char[needed];
    sprintf(buff, gauss_blur_frag_src, sample_from_external, n / 2 + 1, gauss_coeffs.c_str(), pixel_shift.x, pixel_shift.y);
    return buff;
}

void docscanner::texture_downsampler::init(uvec2 input_size, uvec2 output_size, bool input_is_oes_texture, const texture* input_tex, f32 relaxation_factor) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->input_is_oes_texture = input_is_oes_texture;
    this->input_tex = input_tex;

    ASSERT(input_is_oes_texture, "Input must be OES texture.");

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
    
    char* gauss_frag_src_x = prepare_gauss_fragment_src(true, req_kernel_size.x, {1.0f / (f32)input_size.x, 0.0f});
    gauss_blur_x_program = compile_and_link_program(vert_src, gauss_frag_src_x, null, null);
    ASSERT(gauss_blur_x_program.program, "gauss_blur_x_program program could not be compiled.");
    
    char* gauss_frag_src_y = prepare_gauss_fragment_src(false, req_kernel_size.y, {0.0f, 1.0f / (f32)input_size.y});
    gauss_blur_y_program = compile_and_link_program(vert_src, gauss_frag_src_y, null, null);
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

GLuint load_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);

    if (shader) {
        glShaderSource(shader, 1, &source, nullptr);
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

    return shader;
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

shader_program docscanner::compile_and_link_program(const char* vert_src, const char* frag_src, GLuint* vert_out, GLuint* frag_out) {
    GLuint vert_shader = load_shader(GL_VERTEX_SHADER, vert_src);
    if (!vert_shader) {
        LOGE_AND_BREAK("Vertex Shader could not be compiled.");
        return {0};
    }

    GLuint frag_shader = load_shader(GL_FRAGMENT_SHADER, frag_src);
    if (!frag_shader) {
        LOGE_AND_BREAK("Fragment Shader could not be compiled.");
        return {0};
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, vert_shader);
        check_gl_error("glAttachShader");
        glAttachShader(program, frag_shader);
        check_gl_error("glAttachShader");

        link_program(program);

        glDetachShader(program, vert_shader);
        glDeleteShader(vert_shader);
        vert_shader = 0;

        glDetachShader(program, frag_shader);
        glDeleteShader(frag_shader);
        frag_shader = 0;
    }

    if (vert_out) *vert_out = vert_shader;
    if (frag_out) *frag_out = frag_shader;

    return {program};
}

shader_program docscanner::compile_and_link_program(const char* comp_src) {
    GLuint comp_shader = load_shader(GL_COMPUTE_SHADER, comp_src);
    if (!comp_shader) {
        LOGE_AND_BREAK("Compute Shader could not be compiled.");
        return {0};
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, comp_shader);
        check_gl_error("glAttachShader");

        link_program(program);

        glDetachShader(program, comp_shader);
        glDeleteShader(comp_shader);
        comp_shader = 0;
    }

    return {program};
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
    return {glGetUniformLocation(program.program, name)};
}

void docscanner::draw(const canvas &canvas) {
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(canvas.bg_color.x, canvas.bg_color.y, canvas.bg_color.z, 1);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, null);
}

#endif