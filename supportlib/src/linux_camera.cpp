#ifdef LINUX

#include "camera.hpp"
#include "log.hpp"

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

camera docscanner::find_and_open_back_camera(const uvec2& min_size, uvec2& size) {
    camera cam = {};

    cam.fd = open("/dev/video0", O_RDWR);
    
    if(cam.fd < 0) {
        LOGE("Failed to open video device.");
        return {};
    }

    v4l2_capability cap = {};

    if(ioctl(cam.fd, VIDIOC_QUERYCAP, &cap) < 0) {
        LOGE("Failed to query video device capabilities.");
        return {};
    }

    v4l2_format image_format;
    image_format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    image_format.fmt.pix.width = min_size.x;
    image_format.fmt.pix.height = min_size.y;
    image_format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    image_format.fmt.pix.field = V4L2_FIELD_NONE;

    if(ioctl(cam.fd, VIDIOC_S_FMT, &image_format) < 0) {
        LOGE("Failed to set image format.");
        return {};
    }

    if(ioctl(cam.fd, VIDIOC_G_FMT, &image_format) < 0) {
        LOGE("Failed to get image format.");
        return {};
    }
    LOGI("w: %u", image_format.fmt.pix.width);
    LOGI("h: %u", image_format.fmt.pix.height);

    u32 pixel_format = image_format.fmt.pix.pixelformat;

    if(pixel_format != V4L2_PIX_FMT_YUYV) {
        LOGE_AND_BREAK("Webcam format is not supported.");
    }

    cam.cam_size = { image_format.fmt.pix.width, image_format.fmt.pix.height };
    size = cam.cam_size;

    v4l2_requestbuffers req_buffer = {};
    req_buffer.count = 1;
    req_buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req_buffer.memory = V4L2_MEMORY_MMAP;

    if(ioctl(cam.fd, VIDIOC_REQBUFS, &req_buffer) < 0) {
        LOGE("Failed to query video buffer information.");
        return {};
    }

    v4l2_buffer query_buffer = {0};
    query_buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    query_buffer.memory = V4L2_MEMORY_MMAP;
    query_buffer.index = 0;
    if(ioctl(cam.fd, VIDIOC_QUERYBUF, &query_buffer) < 0){
        LOGE("Device failed to return the device information.");
        return {};
    }

    cam.buffer = reinterpret_cast<u8*>(mmap(null, query_buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam.fd, query_buffer.m.offset));
    memset(cam.buffer, 0, query_buffer.length);
    cam.buffer_size = query_buffer.length;

    v4l2_buffer bufferinfo = {};
    bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferinfo.memory = V4L2_MEMORY_MMAP;
    bufferinfo.index = 0;

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(cam.fd, VIDIOC_STREAMON, &type) < 0){
        perror("Could not start streaming, VIDIOC_STREAMON");
        return {};
    }

    if(ioctl(cam.fd, VIDIOC_QBUF, &bufferinfo) < 0){
        LOGE("Could not queue buffer, VIDIOC_QBUF");
        return {};
    }

    cam.f32_buffer = new f32[size.x * size.y * 4];
    cam.cam_tex = create_texture(size, 0x8814);

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

void docscanner::camera::get() {
    v4l2_buffer bufferinfo = {};
    bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferinfo.memory = V4L2_MEMORY_MMAP;
    bufferinfo.index = 0;

    if(ioctl(fd, VIDIOC_DQBUF, &bufferinfo) < 0){
        LOGE("Could not dequeue the buffer, VIDIOC_DQBUF");
        return;
    }

    if(ioctl(fd, VIDIOC_QBUF, &bufferinfo) < 0){
        LOGE("Could not queue buffer, VIDIOC_QBUF");
        return;
    }

    u32 f_i = 0;
    for(u32 i = 0; i < buffer_size; i += 8) {
        u8 Y00 = buffer[i + 0], Cb00 = buffer[i + 1];
        u8 Y01 = buffer[i + 2], Cr00 = buffer[i + 3];
        u8 Y02 = buffer[i + 4], Cb01 = buffer[i + 5];
        u8 Y03 = buffer[i + 6], Cr01 = buffer[i + 7];

        f32 C_R0 = 0.0f, C_G0 = 0.0f, C_B0 = 0.0f;
        f32 C_R1 = 0.0f, C_G1 = 0.0f, C_B1 = 0.0f;
        f32 C_R2 = 0.0f, C_G2 = 0.0f, C_B2 = 0.0f;
        f32 C_R3 = 0.0f, C_G3 = 0.0f, C_B3 = 0.0f;

        RGBFromYCbCr(Y00, Cb00, Cr00, C_R0, C_G0, C_B0);
        RGBFromYCbCr(Y01, Cb00, Cr00, C_R1, C_G1, C_B1);
        RGBFromYCbCr(Y02, Cb01, Cr01, C_R2, C_G2, C_B2);
        RGBFromYCbCr(Y03, Cb01, Cr01, C_R3, C_G3, C_B3);

#define write_out_pixel(i) \
        f32_buffer[f_i + 0] = C_R##i; \
        f32_buffer[f_i + 1] = C_G##i; \
        f32_buffer[f_i + 2] = C_B##i; \
        f32_buffer[f_i + 3] = 1.0;  \
        f_i += 4;

        write_out_pixel(0);
        write_out_pixel(1);
        write_out_pixel(2);
        write_out_pixel(3);
    }

    set_texture_data(cam_tex, reinterpret_cast<u8*>(f32_buffer), cam_size.x, cam_size.y);
}

void docscanner::init_camera_capture(const camera& cam) {
    
}

#endif