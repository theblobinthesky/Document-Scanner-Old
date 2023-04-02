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

pipeline::pipeline() 
    : start_time(0), cam_preview_screen(&backend),
      shutter_animation(&backend, animation_curve::EASE_IN_OUT, 0.75f, 0.65f, 0.0f, 0.15f, RESET_AFTER_COMPLETION | CONTINUE_PLAYING_REVERSED) {}
    
void docscanner::pipeline::pre_init(uvec2 preview_size, int* cam_width, int* cam_height) {
    this->preview_size = preview_size;
    aspect_ratio = preview_size.y / (f32)preview_size.x;

    cam_preview_screen.pre_init(preview_size, cam_width, cam_height);
    backend.input.init(preview_size, aspect_ratio);
}

#ifdef ANDROID
void docscanner::pipeline::init_backend(ANativeWindow* texture_window, file_context* file_ctx) {
#elif defined(LINUX)
void docscanner::pipeline::init_backend() {
#endif
    backend.init(aspect_ratio);

    projection_matrix = mat4::orthographic(0.0f, 1.0f, aspect_ratio, 0.0f, -1.0f, 1.0f);

    f32 cam_preview_bottom_edge = 0.1f;
#ifdef ANDROID
    cam_preview_screen.init_backend(file_ctx, cam_preview_bottom_edge);
    cam_preview_screen.init_cam(texture_window);
#elif defined(LINUX)
    cam_preview_screen.init_backend(null, cam_preview_bottom_edge);
    cam_preview_screen.init_cam();
#endif

    shutter_program = backend.compile_and_link(vert_quad_src, frag_shutter_src);
}

void docscanner::pipeline::render() {
    SCOPED_CAMERA_MATRIX(&backend, projection_matrix);

    get_time(start_time, backend.time);

    cam_preview_screen.render(backend.time);

    vec2 pos = { 0.5f, aspect_ratio - 0.25f };
    vec2 size = { 0.25f, 0.25f };
    backend.use_program(shutter_program);

    motion_event event = backend.input.get_motion_event(pos - size * 0.5f, pos + size * 0.5f);
    if(event.type == motion_type::TOUCH_DOWN) {
#ifdef ANDROID
        pause_camera_capture(cam_preview_screen.cam);
#endif
        cam_preview_screen.unwrap();
        shutter_animation.start();
    }

    get_variable(shutter_program, "inner_out").set_f32(shutter_animation.update());
    backend.draw_quad(pos, size);

    backend.input.end_frame();
}