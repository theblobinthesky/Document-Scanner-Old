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
    aspect_ratio = preview_size.y / (f32)preview_size.x;

    cam_preview_screen.pre_init(preview_size, cam_width, cam_height);
    input.init(preview_size, aspect_ratio);
}

#ifdef ANDROID
void docscanner::pipeline::init_backend(ANativeWindow* texture_window, file_context* file_ctx) {
    backend.init();

    projection_matrix = mat4::orthographic(0.0f, 1.0f, aspect_ratio, 0.0f, -1.0f, 1.0f);

    cam_preview_screen.init_backend(&backend, file_ctx, 0.05f);
    cam_preview_screen.init_cam(texture_window);

    shutter_program = backend.compile_and_link(vert_quad_src, frag_shutter_src);
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

    vec2 pos = { 0.5f, aspect_ratio - 0.25f };
    vec2 size = { 0.25f, 0.25f };
    backend.use_program(shutter_program);

    motion_event event = input.get_motion_event(pos - size * 0.5f, pos + size * 0.5f);
    if(event.type == motion_type::TOUCH_DOWN) {
        anim_start_time = backend.time;
        anim_started = true;
        anim_duration = 0.25f;
    }

    f32 inner_out = 0.75f;

    if(anim_started) {
        f32 t = (backend.time - anim_start_time) / anim_duration;

        if(t > 1.0f) {
            LOGI("anim stopped");
            anim_started = false;
        } else {
            t = 2 * (0.5f - abs(0.5f - ease_in_out_quad(t)));
            inner_out = lerp(0.75f, 0.65f, t);
        }
    }

    get_variable(shutter_program, "inner_out").set_f32(inner_out);
    
    backend.draw_quad(pos, size);

    input.end_frame();
}