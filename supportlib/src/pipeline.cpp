#include "pipeline.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "camera.hpp"
#include "nn.hpp"
#include <chrono>

using namespace docscanner;

void get_time(u64& start_time, f32& time) {
    auto now = std::chrono::high_resolution_clock::now();
    u64 time_long = now.time_since_epoch().count();
    
    if(start_time == 0) {
        start_time = time_long;
        time = 0.0f;
    } else {
        time = (time_long - start_time) / 1000000000.000;
    }
}

void docscanner::pipeline::pre_init(uvec2 preview_size, int* cam_width, int* cam_height) {
    this->preview_size = preview_size;
    cam_preview_screen.pre_init(preview_size, cam_width, cam_height);
}

#ifdef ANDROID
void docscanner::pipeline::init_backend(ANativeWindow* texture_window, file_context* file_ctx) {
    backend.init();

    f32 aspect_ratio = preview_size.y / (f32)preview_size.x;
    projection_matrix = mat4::orthographic(0.0f, 1.0f, 0.0f, aspect_ratio, -1.0f, 1.0f);

    cam_preview_screen.init_backend(&backend, file_ctx, 0.1f);
    cam_preview_screen.init_cam(texture_window);
}
#elif defined(LINUX)
void docscanner::pipeline::init_backend() {
    cam_preview_screen.init_backend(null);
    cam_preview_screen.init_cam();
    get_time(start_time, time);
}
#endif

void docscanner::pipeline::render() {
    SCOPED_CAMERA_MATRIX(&backend, projection_matrix);

    get_time(start_time, time);
    backend.time = time;

    cam_preview_screen.render(time);
}