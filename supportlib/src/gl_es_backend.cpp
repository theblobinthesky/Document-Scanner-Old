#ifdef ANDROID
#include "log.h"
#include "backend.h"
#include <cstdlib>
#include <csignal>

using namespace docscanner;

void check_gl_error(const char* op) {
    for (GLenum error = glGetError(); error; error = glGetError()) {
        LOGE_AND_BREAK("glError with code 0x%04x was triggered by %s().\n", error, op);
    }
}

void docscanner::variable::set_mat4(float* data) {
    glUniformMatrix4fv(location, 1, GL_FALSE, data);
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

u32 docscanner::framebuffer_from_texture(const texture& tex, u32 size) {
    GLint prev_tex;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex);
    
    glBindTexture(GL_TEXTURE_2D, tex.id);

    int width, height, internal_format;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &internal_format);

    int num_channels = 0;
    switch (internal_format)
    {
        case GL_R32F: num_channels = 1; break;
        case GL_RG32F: num_channels = 2; break;
        case GL_RGBA32F: num_channels = 4; break;
        default: LOGE_AND_BREAK("Invalid internal format.");
    }

    ASSERT(size == width * height * num_channels * 4, "The texture size doesn't match the buffer size.");

    glBindTexture(GL_TEXTURE_2D, prev_tex);

    GLuint offscreen_framebuffer;
    glGenFramebuffers(1, &offscreen_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, offscreen_framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.id, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    ASSERT(status == GL_FRAMEBUFFER_COMPLETE, "glCheckFramebufferStatus indicates the Framebuffer is incomplete (error code 0x%x).", status);

    return offscreen_framebuffer;
}

void docscanner::get_framebuffer_data(u32 fb, u8* &data, u32 size) { 
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glReadPixels(0, 0, 128, 128, GL_RGBA, GL_FLOAT, data);
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