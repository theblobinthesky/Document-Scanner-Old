#ifdef LINUX
#include "platform.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "backend.hpp"
#include "pipeline.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <fstream>
#include <string>

using namespace docscanner;

constexpr svec2 size = { 1080 / 2, 2400 / 2 };

struct docscanner::camera {
    s32 fd;
    svec2 cam_size;
    u8* buffer;
    u32 buffer_size;
    f32* f32_buffer;
    f32* rot_f32_buffer;
    texture cam_tex;
};

camera* docscanner::find_and_open_back_camera(const svec2& min_size, svec2& size) {
    camera* cam = new camera();

    printf("find_and_open_camera");
    cam->fd = open("/dev/video0", O_RDWR);
    
    if(cam->fd < 0) {
        LOGE("Failed to open video device.");
        return {};
    }

    v4l2_capability cap = {};

    if(ioctl(cam->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        LOGE("Failed to query video device capabilities.");
        return {};
    }

    v4l2_format image_format;
    image_format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    image_format.fmt.pix.width = min_size.x;
    image_format.fmt.pix.height = min_size.y;
    image_format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    image_format.fmt.pix.field = V4L2_FIELD_NONE;

    if(ioctl(cam->fd, VIDIOC_S_FMT, &image_format) < 0) {
        LOGE("Failed to set image format.");
        return {};
    }

    if(ioctl(cam->fd, VIDIOC_G_FMT, &image_format) < 0) {
        LOGE("Failed to get image format.");
        return {};
    }
    LOGI("w: %u", image_format.fmt.pix.width);
    LOGI("h: %u", image_format.fmt.pix.height);

    u32 pixel_format = image_format.fmt.pix.pixelformat;

    if(pixel_format != V4L2_PIX_FMT_YUYV) {
        LOGE_AND_BREAK("Webcam format is not supported.");
    }

    cam->cam_size = { (s32)image_format.fmt.pix.width, (s32)image_format.fmt.pix.height };
    size = cam->cam_size;

    v4l2_requestbuffers req_buffer = {};
    req_buffer.count = 1;
    req_buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req_buffer.memory = V4L2_MEMORY_MMAP;

    if(ioctl(cam->fd, VIDIOC_REQBUFS, &req_buffer) < 0) {
        LOGE("Failed to query video buffer information.");
        return {};
    }

    v4l2_buffer query_buffer = {};
    query_buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    query_buffer.memory = V4L2_MEMORY_MMAP;
    query_buffer.index = 0;
    if(ioctl(cam->fd, VIDIOC_QUERYBUF, &query_buffer) < 0){
        LOGE("Device failed to return the device information.");
        return {};
    }

    cam->buffer = reinterpret_cast<u8*>(mmap(null, query_buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam->fd, query_buffer.m.offset));
    memset(cam->buffer, 0, query_buffer.length);
    cam->buffer_size = query_buffer.length;

    v4l2_buffer bufferinfo = {};
    bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferinfo.memory = V4L2_MEMORY_MMAP;
    bufferinfo.index = 0;

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(cam->fd, VIDIOC_STREAMON, &type) < 0){
        perror("Could not start streaming, VIDIOC_STREAMON");
        return {};
    }

    if(ioctl(cam->fd, VIDIOC_QBUF, &bufferinfo) < 0){
        LOGE("Could not queue buffer, VIDIOC_QBUF");
        return {};
    }

    cam->rot_f32_buffer = new f32[size.x * size.y * 4];
    cam->f32_buffer = new f32[size.x * size.y * 4];
    cam->cam_tex = make_texture(size, 0x8814);

    return cam;
}

inline void RGBFromYCbCr(s32 y, s32 cb, s32 cr, f32& fr, f32& fg, f32& fb)
{
    s32 r = (s32) (y + 1.40200 * (cr - 0x80));
    s32 g = (s32) (y - 0.34414 * (cb - 0x80) - 0.71414 * (cr - 0x80));
    s32 b = (s32) (y + 1.77200 * (cb - 0x80));  

    fr = ((f32) std::max(0, std::min(255, r))) / 255.0f;
    fg = ((f32) std::max(0, std::min(255, g))) / 255.0f;
    fb = ((f32) std::max(0, std::min(255, b))) / 255.0f;
}

void docscanner::resume_camera_capture(camera* cam) {}

void docscanner::pause_camera_capture(const camera* cam) {}

void docscanner::get_camera_frame(const camera* cam) {
    v4l2_buffer bufferinfo = {};
    bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferinfo.memory = V4L2_MEMORY_MMAP;
    bufferinfo.index = 0;

    if(ioctl(cam->fd, VIDIOC_DQBUF, &bufferinfo) < 0){
        LOGE("Could not dequeue the buffer, VIDIOC_DQBUF");
        return;
    }

    if(ioctl(cam->fd, VIDIOC_QBUF, &bufferinfo) < 0){
        LOGE("Could not queue buffer, VIDIOC_QBUF");
        return;
    }

    u32 f_i = 0;
    for(u32 i = 0; i < cam->buffer_size; i += 8) {
        u8 Y00 = cam->buffer[i + 0], Cb00 = cam->buffer[i + 1];
        u8 Y01 = cam->buffer[i + 2], Cr00 = cam->buffer[i + 3];
        u8 Y02 = cam->buffer[i + 4], Cb01 = cam->buffer[i + 5];
        u8 Y03 = cam->buffer[i + 6], Cr01 = cam->buffer[i + 7];

        f32 C_R0 = 0.0f, C_G0 = 0.0f, C_B0 = 0.0f;
        f32 C_R1 = 0.0f, C_G1 = 0.0f, C_B1 = 0.0f;
        f32 C_R2 = 0.0f, C_G2 = 0.0f, C_B2 = 0.0f;
        f32 C_R3 = 0.0f, C_G3 = 0.0f, C_B3 = 0.0f;

        RGBFromYCbCr(Y00, Cb00, Cr00, C_R0, C_G0, C_B0);
        RGBFromYCbCr(Y01, Cb00, Cr00, C_R1, C_G1, C_B1);
        RGBFromYCbCr(Y02, Cb01, Cr01, C_R2, C_G2, C_B2);
        RGBFromYCbCr(Y03, Cb01, Cr01, C_R3, C_G3, C_B3);


#define write_out_pixel(i) \
        cam->rot_f32_buffer[f_i + 0] = C_R##i; \
        cam->rot_f32_buffer[f_i + 1] = C_G##i; \
        cam->rot_f32_buffer[f_i + 2] = C_B##i; \
        cam->rot_f32_buffer[f_i + 3] = 1.0;  \
        f_i += 4;

        write_out_pixel(0);
        write_out_pixel(1);
        write_out_pixel(2);
        write_out_pixel(3);
    }

    set_texture_data(cam->cam_tex, reinterpret_cast<u8*>(cam->rot_f32_buffer), cam->cam_size); // todo: texture is janky
}

const texture* docscanner::get_camera_frame_texture(const camera* cam) {
    return &cam->cam_tex;
}

void read_from_file(const char* path, u8* &data, u32 &size) {
    FILE* file = fopen(path, "rb");
    ASSERT(file, "fopen(\"%s\", \"rb\") failed.", path);

    fseek(file, 0, SEEK_END);
    size = (u32)ftell(file);
    rewind(file);

    data = new u8[size];
    u32 read = (u32)fread(data, 1, size, file);
    ASSERT(read == size, "fread failed. read %u, but expected %u.", read, size);
}

void write_to_file(const char* path, u8* data, u32 size) {
    FILE* file = fopen(path, "wb");
    ASSERT(file, "fopen(\"%s\", \"wb\") failed.", path);
    
    u32 wrote = (u32)fwrite(data, 1, size, file);
    ASSERT(wrote == size, "fwrite failed. wrote %u, but expected %u.", wrote, size);
    
    fclose(file);
}

void docscanner::read_from_package(file_context* ctx, const char* path, u8* &data, u32 &size) {
    read_from_file(path, data, size);
}

void docscanner::read_from_internal_file(file_context* ctx, const char* path, u8* &data, u32 &size) {
    read_from_file(path, data, size);
}

void docscanner::write_to_internal_file(file_context* ctx, const char* path, u8* data, u32 size) {
    write_to_file(path, data, size);
}

int platform_init() {
    if(!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW.\n");
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window;
    window = glfwCreateWindow(size.x, size.y, "Document Scanner", NULL, NULL);
    if(!window) {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize OpenGL context.\n");
        return -1;
    }
    
    thread_pool threads;
    asset_manager* assets = new asset_manager(null, "test.assetpack", &threads);

    svec2 cam_size;
    camera* cam = find_and_open_back_camera(size, cam_size);
    resume_camera_capture(cam);

    pipeline_args args = {
        .texture_window = null, .assets = assets, .preview_size = size,
        .enable_dark_mode = false, .threads = &threads
    };

    pipeline pipe(args);
    pipe.init_camera_related(cam, cam_size);
    pipe.backend.cam_is_init = true;

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    do {
        int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        if (state == GLFW_PRESS) {
            double x, y;
            glfwGetCursorPos(window, &x, &y);

            motion_event event = {
                .type = docscanner::motion_type::TOUCH_DOWN,
                .pos = { (f32)x, (f32)y }
            };

            pipe.backend.input.handle_motion_event(event);
        }
        
        pipe.render();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(0);

    return 0;
}

int main() {
    return platform_init();
}

#endif