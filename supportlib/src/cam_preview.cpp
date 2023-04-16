#include "cam_preview.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "input.hpp"
#include "camera.hpp"
#include <chrono>

using namespace docscanner;

constexpr f32 preview_aspect_ratio = 16.0f / 9.0f;
constexpr svec2 unwrap_size = { 1240, 1754 };
constexpr svec2 downsampled_size = {64, 64};
constexpr rect cam_preview_crad = { { 0.1, 0.1 }, { 0.1, 0.1 } };

cam_preview::cam_preview(engine_backend* backend, ui_manager* ui, f32 bottom_edge, const rect& unwrapped_rect) 
    : backend(backend), ui(ui),
      shutter_animation(backend, animation_curve::EASE_IN_OUT, 0.75f, 0.65f, 0.0f, 0.15f, RESET_AFTER_COMPLETION | CONTINUE_PLAYING_REVERSED),
      unwrap_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 1.0f, 0),
      blendout_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.2f, 0.5f, 0),
      is_live_camera_streaming(true), is_init(false) {
    SCOPED_TIMER("init_backend");

    this->bottom_edge = bottom_edge;
    this->unwrapped_rect = unwrapped_rect;

    nn_input_buffer_size = downsampled_size.area() * 4 * sizeof(f32);
    nn_input_buffer = new u8[nn_input_buffer_size];

    nn_contour_out_size = downsampled_size.area() * points_per_contour * sizeof(f32);
    nn_contour_out = new u8[nn_contour_out_size];

    shutter_program = backend->compile_and_link(vert_quad_src(), frag_shutter_src());
    
    create_neural_network_from_path(backend->assets, backend->threads, "contour_model.tflite", execution_pref::sustained_speed);

    is_init = true;
}

void cam_preview::init_camera_related() {
    f32 w = (1.0f / preview_aspect_ratio) * (backend->cam_size_px.x / (f32)backend->cam_size_px.y);
    f32 l = 0.5f - w / 2.0f;
    f32 r = 0.5f + w / 2.0f;

    cam_uv_bounds = {
        .tl = { l, 0 },
        .br = { r, 1 }
    };

    f32 pb = backend->preview_height - bottom_edge;
    f32 pt = pb - preview_aspect_ratio;

    cam_pos_bounds = {
        .tl = { 0, pt },
        .br = { 1, pb }
    };


    mesher.init(&nn_exists_out, (f32*)nn_contour_out, downsampled_size, cam_uv_bounds, cam_pos_bounds, 0.4f);

    particles.init(backend, &mesher, svec2({ 4, 4 }), 0.05f, 0.015f, 2.0f);
    border.init(backend, &mesher, svec2({ 16, 16 }), 0.01f);
    cutout.init(backend, &mesher);

    unwrapped_vertices = new vertex[mesher.mesh_size.area()];
    for(s32 x = 0; x < mesher.mesh_size.x; x++) {
        for(s32 y = 0; y < mesher.mesh_size.y; y++) {
            s32 i = x * mesher.mesh_size.y + y;
            f32 x_t = x / (f32)(mesher.mesh_size.x - 1);
            f32 y_t = y / (f32)(mesher.mesh_size.y - 1);

            mesher.blend_to_vertices[i] = { 
                lerp(unwrapped_rect.tl.x, unwrapped_rect.br.x, x_t), 
                lerp(unwrapped_rect.tl.y, unwrapped_rect.br.y, 1.0f - y_t) // todo: fix these coordinate system inconsistencies
            };
        }
    }
    
#if CAM_USES_OES_TEXTURE
    tex_downsampler.init(backend, backend->cam_size_px, downsampled_size, true, null, 2, 1.0f);
    tex_sampler.init(backend, unwrap_size, true, null, 
        mesher.blend_vertices, mesher.mesh_size.area(), mesher.mesh_indices.data(), mesher.mesh_indices.size());
#else
    tex_downsampler.init(backend, backend->cam_size_px, downsampled_size, false, &backend->cam.cam_tex, 2, 1.0f);
    tex_sampler.init(backend, unwrap_size, false, &backend->cam.cam_tex, 
        mesher.blend_vertices, mesher.mesh_size.area(), mesher.mesh_indices.data(), mesher.mesh_indices.size());
#endif


#if CAM_USES_OES_TEXTURE
    tex_sampler.init(backend, backend->cam_size_px, true, null, unwrapped_vertices, mesher.mesh_size.area(), mesher.mesh_indices.data(), mesher.mesh_indices.size());
#else
    tex_sampler.init(backend, backend->cam_size_px, false, &backend->cam.cam_tex, unwrapped_vertices, mesher.mesh_size.area(), mesher.mesh_indices.data(), mesher.mesh_indices.size());
#endif
}

void cam_preview::unwrap() {
    is_live_camera_streaming = false;

#ifdef ANDROID
    pause_camera_capture(backend->cam);
#endif

    for(s32 x = 0; x < mesher.mesh_size.x; x++) {
        for(s32 y = 0; y < mesher.mesh_size.y; y++) {
            s32 i = x * mesher.mesh_size.y + y;
            f32 x_t = x / (f32)(mesher.mesh_size.x - 1);
            f32 y_t = y / (f32)(mesher.mesh_size.y - 1);
            unwrapped_vertices[i] = {
                .pos = { lerp(0, 1, x_t), lerp(0, 1, 1.0f - y_t) },
                .uv = mesher.blend_vertices[i].uv
            };
        }
    }

    tex_sampler.sample();
    unbind_framebuffer();

    unwrap_animation.start();
    blendout_animation.start();
}

void docscanner::cam_preview::render(f32 time) {
    if(!is_init) return;

    if(is_live_camera_streaming && backend->cam_is_init) {
        backend->cam.get();

        tex_downsampler.downsample();

        get_framebuffer_data(*tex_downsampler.output_fb, tex_downsampler.output_size, nn_input_buffer, nn_input_buffer_size);
        
        constexpr u32 out_size = 1; // 2;
        u8* out_datas[out_size] = { nn_contour_out }; //, (u8*)&nn_exists_out };
        u32 out_sizes[out_size] = { nn_contour_out_size }; // , sizeof(f32) };
        invoke_neural_network_on_data(backend->assets, nn_input_buffer, nn_input_buffer_size, out_datas, out_sizes, out_size);

        mesher.mesh(backend);
    }

    if(backend->cam_is_init) {
        mesher.blend(unwrap_animation.update());
        blendout_animation.update();
    }

    if(backend->cam_is_init) {
        unbind_framebuffer();
        prepare_empty_canvas(ui->theme.black);

#if CAM_USES_OES_TEXTURE
        backend->draw_rounded_oes_textured_quad(cam_pos_bounds, cam_preview_crad, cam_uv_bounds, rot_mode::ROT_270_DEG);
#else
    //backend->cam.cam_tex
#endif
    }

    vec2 pos = { 0.5f, backend->preview_height - 0.25f };
    vec2 size = { 0.25f, 0.25f };
    rect shutter_rect = rect::from_middle_and_size(pos, size);

    motion_event event = backend->input.get_motion_event(shutter_rect);
    if(event.type == motion_type::TOUCH_DOWN) {
        shutter_animation.start();
        unwrap();
    }

    if(backend->cam_is_init) {
        cutout.render(time);
        particles.render(backend);
        border.render(time);
    }
    
    backend->use_program(shutter_program);
    get_variable(shutter_program, "opacity").set_f32(lerp(1.0f, 0.0f, blendout_animation.value));

    get_variable(shutter_program, "inner_out").set_f32(shutter_animation.update());
    backend->draw_quad(shutter_program, shutter_rect);

    backend->DEBUG_draw();
}