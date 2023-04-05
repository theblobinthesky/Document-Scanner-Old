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
    discard_button(ui, "Discard", crad_thin_to_the_left, ui->theme.primary_color), next_button(ui, "Keep", crad_thin_to_the_right, ui->theme.primary_color),
    bg_blendin_animation(ui->backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 1.0f, 0), 
    fg_blendin_animation(ui->backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.2f, 1.0f, 0),
    split_animation(ui->backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 1.0f, 0) {
    top_unwrapped_rect = grid_split(unwrapped_rect, 0, 2, split_direction::VERTICAL);
    bottom_unwrapped_rect = grid_split(unwrapped_rect, 1, 2, split_direction::VERTICAL);

    rect screen = {
        .tl = {},
        .br = { 1, ui->backend->preview_height }
    };

    screen = get_between(screen, 0.8f, 0.9f);

    rect left_right_margin = { { 0.02f, 0 }, { 0.02f, 0 } };
    discard_button.layout(cut_margins(grid_split(screen, 0, 2, split_direction::HORIZONTAL), left_right_margin));
    next_button.layout(cut_margins(grid_split(screen, 1, 2, split_direction::HORIZONTAL), left_right_margin));

    sampler_program = ui->backend->compile_and_link(vert_quad_src(), frag_simple_tex_sampler_src(false, 0));

    vec2 unwrapped_size = unwrapped_rect.size();
    vec2* border_points = new vec2[4] {
        unwrapped_rect.tl, unwrapped_rect.tl + vec2({ unwrapped_size.x, 0 }),
        unwrapped_rect.br, unwrapped_rect.tl + vec2({ 0, unwrapped_size.y })
    };

    f32 hl_border_perc = 0.1f;
    vec2 hl_border_size_x = { hl_border_perc, 0 };
    vec2 hl_border_size_y = { 0, hl_border_perc };

    vec2* tl_border_points = new vec2[3] {
        unwrapped_rect.tl + hl_border_size_y, unwrapped_rect.tl, unwrapped_rect.tl + hl_border_size_x
    };

    vec2 unwrapped_rect_tr = unwrapped_rect.tl + vec2({ unwrapped_size.x, 0 });    
    vec2* tr_border_points = new vec2[3] {
        unwrapped_rect_tr - hl_border_size_x, unwrapped_rect_tr, unwrapped_rect_tr + hl_border_size_y
    };

    vec2* br_border_points = new vec2[3] {
        unwrapped_rect.br - hl_border_size_y, unwrapped_rect.br, unwrapped_rect.br - hl_border_size_x
    };

    vec2 unwrapped_rect_bl = unwrapped_rect.tl + vec2({ 0, unwrapped_size.y });
    vec2* bl_border_points = new vec2[3] {
        unwrapped_rect_bl - hl_border_size_y, unwrapped_rect_bl, unwrapped_rect_bl + hl_border_size_x
    };

    border_lines.init(ui->backend, border_points, 4, 0.01f, ui->theme.primary_color, true);
    tl_lines.init(ui->backend, tl_border_points, 3, 0.03f, ui->theme.primary_dark_color, false);
    tr_lines.init(ui->backend, tr_border_points, 3, 0.03f, ui->theme.primary_dark_color, false);
    br_lines.init(ui->backend, br_border_points, 3, 0.03f, ui->theme.primary_dark_color, false);
    bl_lines.init(ui->backend, bl_border_points, 3, 0.03f, ui->theme.primary_dark_color, false);
}

void unwrapped_options_screen::draw() {
    if(bg_blendin_animation.state == animation_state::WAITING) {
        bg_blendin_animation.start();
        fg_blendin_animation.start();
    }

    canvas c = { vec3::lerp(ui->theme.black, ui->theme.background_color, bg_blendin_animation.update()) };
    ::draw(c);

    bind_texture_to_slot(0, *unwrapped_texture);



    split_animation.update();

    ui->backend->use_program(sampler_program);
    get_variable(sampler_program, "saturation").set_f32(1.0f);
    get_variable(sampler_program, "opacity").set_f32(1.0f);

    rect unwrapped_uv = { {}, { 1, 1 } };
    rect split_unwrapped_uv = { {}, { 1, 0.5f } };

    if(split_animation.state != animation_state::WAITING) {
        ui->backend->draw_quad(sampler_program,
            rect::from_middle_and_size(unwrapped_rect.tl, unwrapped_rect.br),// rect::lerp(unwrapped_rect, bottom_unwrapped_rect, split_animation.value),
            rect::lerp(unwrapped_uv, split_unwrapped_uv, split_animation.value));
    }

    ui->backend->draw_quad(sampler_program, 
            rect::lerp(unwrapped_rect, top_unwrapped_rect, split_animation.value),
            rect::lerp(unwrapped_uv, split_unwrapped_uv, split_animation.value));


    border_lines.color.w = fg_blendin_animation.update();
    tl_lines.color.w = fg_blendin_animation.value;
    tr_lines.color.w = fg_blendin_animation.value;
    br_lines.color.w = fg_blendin_animation.value;
    bl_lines.color.w = fg_blendin_animation.value;

    border_lines.draw();
    tl_lines.draw();
    tr_lines.draw();
    br_lines.draw();
    bl_lines.draw();

    discard_button.color.w = fg_blendin_animation.value;
    next_button.color.w = fg_blendin_animation.value;

    if(discard_button.draw()) {
        LOGI("discard!");
    }

    if(next_button.draw()) {
        split_animation.start();
    }
}

camera* docscanner::pipeline::pre_init(svec2 preview_size, svec2& cam_size) {
    camera* cam = new camera();
    *cam = find_and_open_back_camera(preview_size, cam_size);
    return cam;
}

pipeline::pipeline(const pipeline_args& args)
    : backend(args.preview_size, args.cam_size, args.file_ctx), ui(&backend, args.enable_dark_mode), 
    cam_preview_screen(&backend, &ui, args.cam), options_screen(&ui, unwrapped_mesh_rect, &cam_preview_screen.tex_sampler.output_tex),
    start_time(0), last_time(0) {
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

    if(cam_preview_screen.blendout_animation.state == FINISHED) {
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