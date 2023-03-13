#include "pipeline.h"
#include "log.h"
#include "backend.h"
#include "android_camera.h"
#include "nn.h"

using namespace docscanner;

void docscanner::pipeline::pre_init(int* cam_width, int* cam_height) {
    cam_preview_screen.pre_init(cam_width, cam_height);
}

void docscanner::pipeline::init_backend(ANativeWindow* texture_window, uvec2 preview_size, file_context* file_ctx) {
    cam_preview_screen.init_backend(preview_size, file_ctx);
    cam_preview_screen.init_cam(texture_window);
}

void docscanner::pipeline::render() {
    cam_preview_screen.render();
}