#include "pipeline.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "camera.hpp"
#include "nn.hpp"
#include <chrono>

using namespace docscanner;

constexpr f32 paper_aspect_ratio = 1.41421356237;

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

unwrapped_options_screen::unwrapped_options_screen(ui_manager* ui) : ui(ui) {}

void unwrapped_options_screen::init(const rect& unwrapped_rect) {
    this->unwrapped_rect = unwrapped_rect;
    discard_button.init(ui, unwrapped_rect, "Discard");
    next_button.init(ui, unwrapped_rect, "Keep");
}

void unwrapped_options_screen::draw() {
    canvas c = { ui->theme.background_color };
    ::draw(c);
    discard_button.draw();
    next_button.draw();
}

pipeline::pipeline()
    : ui(&backend), start_time(0), cam_preview_screen(&backend, &ui), options_screen(&ui) {}
    
void docscanner::pipeline::pre_init(svec2 preview_size, int* cam_width, int* cam_height) {
    this->preview_size = preview_size;
    aspect_ratio = preview_size.y / (f32)preview_size.x;

    cam_preview_screen.pre_init(preview_size, cam_width, cam_height);
    backend.input.init(preview_size, aspect_ratio);
}

#ifdef ANDROID
void docscanner::pipeline::init_backend(ANativeWindow* texture_window, file_context* file_ctx, bool enable_dark_mode) {
    backend.file_ctx = file_ctx;
#elif defined(LINUX)
void docscanner::pipeline::init_backend(bool enable_dark_mode) {
#endif
    backend.init(preview_size, aspect_ratio);
    ui.theme.init(enable_dark_mode);

    projection_matrix = mat4::orthographic(0.0f, 1.0f, aspect_ratio, 0.0f, -1.0f, 1.0f);

    f32 cam_preview_bottom_edge = 0.1f;
    
    f32 unwrap_top_border = 0.3f;
    f32 margin = 0.1f;
    f32 uw = 1.0f - 2.0f * margin;
    rect unwrapped_mesh_rect = {
        { margin, uw * unwrap_top_border }, 
        { 1.0f - margin, uw * (unwrap_top_border + paper_aspect_ratio) }
    };

#ifdef ANDROID
    cam_preview_screen.init_backend(file_ctx, cam_preview_bottom_edge, unwrapped_mesh_rect);
    cam_preview_screen.init_cam(texture_window);
#elif defined(LINUX)
    cam_preview_screen.init_backend(null, cam_preview_bottom_edge, unwrapped_mesh_rect);
    cam_preview_screen.init_cam();
#endif

    options_screen.init(unwrapped_mesh_rect);
}

void docscanner::pipeline::render() {
    SCOPED_CAMERA_MATRIX(&backend, projection_matrix);

    get_time(start_time, backend.time);

    if(cam_preview_screen.blendin_animation.state == FINISHED) {
        options_screen.draw();
    } else {
        cam_preview_screen.render(backend.time);
    }

    backend.input.end_frame();

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    auto dur = end - last_time;
    LOGI("frame time: %ums, fps: %u", (u32)dur, (u32)(1000.0f / dur));
    last_time = end;
}