#include "pipeline.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "camera.hpp"
#include "nn.hpp"

using namespace docscanner;

void docscanner::pipeline::pre_init(uvec2 preview_size, int* cam_width, int* cam_height) {
    cam_preview_screen.pre_init(preview_size, cam_width, cam_height);
}

#ifdef ANDROID
void docscanner::pipeline::init_backend(ANativeWindow* texture_window, file_context* file_ctx) {
    cam_preview_screen.init_backend(file_ctx);
    cam_preview_screen.init_cam(texture_window);
}
#elif defined(LINUX)
void docscanner::pipeline::init_backend() {
    cam_preview_screen.init_backend(null);
    cam_preview_screen.init_cam();
}
#endif

void docscanner::pipeline::render() {
    cam_preview_screen.render();
}