#include "cam_preview.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "input.hpp"
#include "camera.hpp"
#include <chrono>

using namespace docscanner;

constexpr f32 preview_aspect_ratio = 16.0f / 9.0f;

cam_preview::cam_preview(engine_backend* backend, ui_manager* ui, camera* cam) 
    : backend(backend), ui(ui), cam(*cam),
      unwrap_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 1.0f, 0),
      blendout_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 0.5f, 0),
      blendin_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 1.0f, 0.5f, 0),
      shutter_animation(backend, animation_curve::EASE_IN_OUT, 0.75f, 0.65f, 0.0f, 0.15f, RESET_AFTER_COMPLETION | CONTINUE_PLAYING_REVERSED),
      is_live_camera_streaming(true) {}

void docscanner::cam_preview::init_backend(f32 bottom_edge, const rect& unwrapped_rect) {
    f32 w = (1.0f / preview_aspect_ratio) * (backend->cam_size_px.x / (f32)backend->cam_size_px.y);
    f32 l = 0.5f - w / 2.0f;
    f32 r = 0.5f + w / 2.0f;

    preview_program = backend->compile_and_link(vert_src, frag_simple_tex_sampler_src(CAM_USES_OES_TEXTURE, 0));

    // buffer stuff
    f32 pb = backend->preview_height - bottom_edge;
    f32 pt = pb - preview_aspect_ratio;

    rect point_range = {
        .tl = { l, 0 },
        .br = { r, 1 }
    };

    rect point_dst = {
        .tl = { 0, pt },
        .br = { 1, pb }
    };

    vertex vertices[] = {
        {{1.f, pb}, {1, l}},
        {{0.f, pt}, {0, r}},
        {{1.f, pt}, {0, l}},
        {{0.f, pb}, {1, r}}
    };

    u32 indices[] = { 
        0, 1, 2, 
        0, 3, 1 
    };

    cam_quad_buffer = make_shader_buffer();
    fill_shader_buffer(cam_quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    svec2 downsampled_size = {64, 64};

    nn_input_buffer_size = downsampled_size.area() * 4 * sizeof(f32);
    nn_input_buffer = new u8[nn_input_buffer_size];

    nn_contour_out_size = downsampled_size.area() * points_per_contour * sizeof(f32);
    nn_contour_out = new u8[nn_contour_out_size];

#if CAM_USES_OES_TEXTURE
    tex_downsampler.init(backend, backend->cam_size_px, downsampled_size, true, null, 2, 1.0f);
#else
    tex_downsampler.init(backend, backend->cam_size_px, downsampled_size, false, &cam.cam_tex, 2, 1.0f);
#endif

    mesher.init(&nn_exists_out, (f32*)nn_contour_out, downsampled_size, point_range, point_dst, 0.4f);

    for(s32 x = 0; x < mesher.mesh_size.x; x++) {
        for(s32 y = 0; y < mesher.mesh_size.y; y++) {
            f32 x_t = x / (f32)(mesher.mesh_size.x - 1);
            f32 y_t = y / (f32)(mesher.mesh_size.y - 1);

            mesher.blend_to_vertices[x * mesher.mesh_size.y + y] = { 
                lerp(unwrapped_rect.tl.x, unwrapped_rect.br.x, x_t), 
                lerp(unwrapped_rect.tl.y, unwrapped_rect.br.y, y_t) // todo: fix these coordinate system inconsitencies
            };
        }
    }

    particles.init(backend, &mesher, svec2({ 4, 4 }), 0.05f, 0.015f, 2.0f);
    border.init(backend, &mesher, svec2({ 16, 16 }), 0.01f);
    cutout.init(backend, &mesher);
    
    shutter_program = backend->compile_and_link(vert_quad_src, frag_shutter_src);
    
    nn = create_neural_network_from_path(backend->file_ctx, "contour_model.tflite", execution_pref::sustained_speed);

    is_init = true;
}

#ifdef ANDROID
void docscanner::cam_preview::init_cam(ANativeWindow* texture_window) {
    init_camera_capture(cam, texture_window);
}
#elif defined(LINUX)
void docscanner::cam_preview::init_cam() {
    init_camera_capture(cam);
}
#endif

void cam_preview::unwrap() {
    is_live_camera_streaming = false;

#ifdef ANDROID
        pause_camera_capture(cam);
#endif

    unwrap_animation.start();
    blendout_animation.start();
    blendin_animation.start();
}

void docscanner::cam_preview::render(f32 time) {
    if(!is_init) return;

    if(is_live_camera_streaming) {
        cam.get();

        tex_downsampler.downsample();

        get_framebuffer_data(*tex_downsampler.output_fb, tex_downsampler.output_size, nn_input_buffer, nn_input_buffer_size);
        
        constexpr u32 out_size = 1; // 2;
        u8* out_datas[out_size] = { nn_contour_out }; //, (u8*)&nn_exists_out };
        u32 out_sizes[out_size] = { nn_contour_out_size }; // , sizeof(f32) };
        invoke_neural_network_on_data(nn, nn_input_buffer, nn_input_buffer_size, out_datas, out_sizes, out_size);

        mesher.mesh(backend);
    }

    mesher.blend(unwrap_animation.update());

    blendout_animation.update();

    canvas c = {
        .bg_color=vec3::lerp(ui->theme.black, ui->theme.background_color, blendin_animation.update())
    };

    backend->use_program(preview_program);
    get_variable(preview_program, "saturation").set_f32(lerp(0.2f, 0.0f, blendout_animation.value));
    get_variable(preview_program, "opacity").set_f32(lerp(0.5f, 0.0f, blendout_animation.value));

    unbind_framebuffer();
    bind_shader_buffer(cam_quad_buffer);
    
#if !CAM_USES_OES_TEXTURE
    bind_texture_to_slot(0, cam.cam_tex);
#endif

    draw(c);

    cutout.render(time);
    particles.render(backend);
    border.render(time);
    
    vec2 pos = { 0.5f, backend->preview_height - 0.25f };
    vec2 size = { 0.25f, 0.25f };

    motion_event event = backend->input.get_motion_event(pos - size * 0.5f, pos + size * 0.5f);
    if(event.type == motion_type::TOUCH_DOWN) {
        shutter_animation.start();
        unwrap();
    }

    backend->use_program(shutter_program);
    get_variable(shutter_program, "opacity").set_f32(lerp(1.0f, 0.0f, blendin_animation.value));

    get_variable(shutter_program, "inner_out").set_f32(shutter_animation.update());
    backend->draw_quad(shutter_program, pos, size);

    backend->DEBUG_draw();
}