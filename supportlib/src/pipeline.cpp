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
    get_time(start_time, time);
}
#endif

void docscanner::pipeline::render() {
    get_time(start_time, time);
    LOGI("time: %f", time);
    cam_preview_screen.render(time);
}