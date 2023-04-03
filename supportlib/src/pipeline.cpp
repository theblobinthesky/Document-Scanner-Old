#include "pipeline.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "camera.hpp"
#include "nn.hpp"
#include <chrono>

using namespace docscanner;

constexpr f32 paper_aspect_ratio = 1.41421356237;

constexpr f32 cam_preview_bottom_edge = 0.1f;

constexpr f32 unwrap_top_border = 0.3f;
constexpr f32 margin = 0.05f;
constexpr f32 uw = 1.0f - 2.0f * margin;
constexpr rect unwrapped_mesh_rect = {
    { margin, uw * unwrap_top_border }, 
    { 1.0f - margin, uw * (unwrap_top_border + paper_aspect_ratio) }
};

constexpr vec2 min_max_button_crad = { 0.05f, 0.1f };
constexpr rect crad_thin_to_the_left = { {min_max_button_crad.x, min_max_button_crad.x}, {min_max_button_crad.y, min_max_button_crad.y} };
constexpr rect crad_thin_to_the_right = { {min_max_button_crad.y, min_max_button_crad.y}, {min_max_button_crad.x, min_max_button_crad.x} };

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

unwrapped_options_screen::unwrapped_options_screen(ui_manager* ui, const rect& unwrapped_rect, const texture* unwrapped_texture) 
    : ui(ui), unwrapped_rect(unwrapped_rect), unwrapped_texture(unwrapped_texture),
    discard_button(ui, "Discard", crad_thin_to_the_left, ui->theme.primary_color), next_button(ui, "Keep", crad_thin_to_the_right, ui->theme.primary_color) {
    rect screen = {
        .tl = {},
        .br = { 1, ui->backend->preview_height }
    };

    screen = get_between(screen, 0.8f, 0.9f);

    rect left_right_margin = { { 0.02f, 0 }, { 0.02f, 0 } };
    discard_button.layout(cut_margins(grid_split_x(screen, 0, 2), left_right_margin));
    next_button.layout(cut_margins(grid_split_x(screen, 1, 2), left_right_margin));

    sampler_program = ui->backend->compile_and_link(vert_quad_src(), frag_simple_tex_sampler_src(false, 0));
}

void unwrapped_options_screen::draw() {
    canvas c = { ui->theme.background_color };
    ::draw(c);

    bind_texture_to_slot(0, *unwrapped_texture);
        
    ui->backend->use_program(sampler_program);
    get_variable(sampler_program, "saturation").set_f32(1.0f);
    get_variable(sampler_program, "opacity").set_f32(1.0f);
    ui->backend->draw_quad(sampler_program, unwrapped_rect.middle(), unwrapped_rect.size());

    discard_button.draw();
    next_button.draw();
}

camera* docscanner::pipeline::pre_init(svec2 preview_size, svec2& cam_size) {
    camera* cam = new camera();
    *cam = find_and_open_back_camera(preview_size, cam_size);
    return cam;
}

pipeline::pipeline(const pipeline_args& args)
    : backend(args.preview_size, args.cam_size, args.file_ctx), ui(&backend, args.enable_dark_mode), start_time(0), 
    cam_preview_screen(&backend, &ui, args.cam), options_screen(&ui, unwrapped_mesh_rect, &cam_preview_screen.tex_sampler.output_tex) {
    projection_matrix = mat4::orthographic(0.0f, 1.0f, backend.preview_height, 0.0f, -1.0f, 1.0f);

#ifdef ANDROID
    cam_preview_screen.init_backend(cam_preview_bottom_edge, unwrapped_mesh_rect);
    cam_preview_screen.init_cam(args.texture_window);
#elif defined(LINUX)
    cam_preview_screen.init_backend(cam_preview_bottom_edge, unwrapped_mesh_rect);
    cam_preview_screen.init_cam();
#endif
}

void docscanner::pipeline::render() {
    SCOPED_CAMERA_MATRIX(&backend, projection_matrix);

    get_time(start_time, backend.time);

    cam_preview_screen.tex_sampler.sample();
    unbind_framebuffer();
    
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