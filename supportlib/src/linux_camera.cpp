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
    image_format.fmt.pix.pixelformat = V4L2_PIX_FMT_ARGB32;
    image_format.fmt.pix.field = V4L2_FIELD_NONE;

    cam.cam_size = min_size;
    size = cam.cam_size;

    if(ioctl(cam.fd, VIDIOC_S_FMT, &image_format) < 0) {
        LOGE("Failed to set image format.");
        return {};
    }

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

    
    v4l2_buffer bufferinfo;
    memset(&bufferinfo, 0, sizeof(bufferinfo));
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

    cam.cam_tex = create_texture(size, GL_RGBA8);

    return cam;
}

void docscanner::camera::get() {
    v4l2_buffer bufferinfo;
    memset(&bufferinfo, 0, sizeof(bufferinfo));
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

    set_texture_data(cam_tex, buffer, cam_size.x, cam_size.y);
}

void docscanner::init_camera_capture(const camera& cam) {
    
}

#endif