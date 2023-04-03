#include "cam_preview.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "input.hpp"
#include "camera.hpp"
#include <chrono>

using namespace docscanner;

constexpr f32 preview_aspect_ratio = 4.0f / 3.0f;
constexpr f32 paper_aspect_ratio = 1.41421356237;

cam_preview::cam_preview(engine_backend* backend) 
    : backend(backend), 
      unwrap_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 1.0f, 0),
      blendout_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 0.0f, 0.5f, 0),
      blendin_animation(backend, animation_curve::EASE_IN_OUT, 0.0f, 1.0f, 1.0f, 0.5f, 0),
      is_live_camera_streaming(true) {}

void docscanner::cam_preview::pre_init(svec2 preview_size, int* cam_width, int* cam_height) {
    this->preview_size = preview_size;

    cam = find_and_open_back_camera(preview_size, cam_tex_size);
    *cam_width = (int) cam_tex_size.x;
    *cam_height = (int) cam_tex_size.y;
}

void docscanner::cam_preview::init_backend(file_context* file_ctx, f32 bottom_edge) {
    f32 aspect_ratio = 16.0f / 9.0f;
    f32 w = (1.0f / aspect_ratio) * (cam_tex_size.x / (f32)cam_tex_size.y);
    f32 l = 0.5f - w / 2.0f;
    f32 r = 0.5f + w / 2.0f;

    preview_program = backend->compile_and_link(vert_src, frag_simple_tex_sampler_src(CAM_USES_OES_TEXTURE, 0));

    // buffer stuff
    f32 pb = backend->preview_height - bottom_edge;
    f32 pt = pb - aspect_ratio;

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
    tex_downsampler.init(backend, { (s32)cam_tex_size.x, (s32)cam_tex_size.y }, downsampled_size, true, null, 2, 1.0f);
#else
    tex_downsampler.init(backend, { (s32)cam_tex_size.x, (s32)cam_tex_size.y }, downsampled_size, false, &cam.cam_tex, 2, 1.0f);
#endif

    mesher.init(&nn_exists_out, (f32*)nn_contour_out, downsampled_size, point_range, point_dst, 0.4f);

    f32 unwrap_top_border = 0.3f;
    f32 margin = 0.1f;
    f32 ul = margin, ur = 1.0f - margin;
    f32 uw = ur - ul;
    f32 ut = uw * unwrap_top_border, ub = uw * (unwrap_top_border + paper_aspect_ratio);

    for(s32 x = 0; x < mesher.mesh_size.x; x++) {
        for(s32 y = 0; y < mesher.mesh_size.y; y++) {
            f32 x_t = x / (f32)(mesher.mesh_size.x - 1);
            f32 y_t = y / (f32)(mesher.mesh_size.y - 1);

            mesher.blend_to_vertices[x * mesher.mesh_size.y + y] = { 
                lerp(ul, ur, x_t), 
                lerp(ub, ut, y_t) // todo: fix these coordinate system inconsitencies
            };
        }
    }

    particles.init(backend, &mesher, svec2({ 4, 4 }), 0.05f, 0.015f, 2.0f);
    border.init(backend, &mesher, svec2({ 16, 16 }), 0.01f);
    cutout.init(backend, &mesher);
    
    nn = create_neural_network_from_path(file_ctx, "contour_model.tflite", execution_pref::sustained_speed);

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
        .bg_color=vec3::lerp({0, 0, 0}, {0.15f, 0.15f, 0.15f}, blendin_animation.update())
    };

    backend->use_program(preview_program);
    get_variable(preview_program, "alpha").set_f32(lerp(0.5f, 0.0f, blendout_animation.value));

    unbind_framebuffer();
    bind_shader_buffer(cam_quad_buffer);
    
#if !CAM_USES_OES_TEXTURE
    bind_texture_to_slot(0, cam.cam_tex);
#endif

    draw(c);

    cutout.render(time);
    particles.render(backend);
    border.render(time);

    backend->DEBUG_draw();

    // auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    // auto dur = end - last_time;
    // LOGI("frame time: %ums, fps: %u", (u32)dur, (u32)(1000.0f / dur));
    // last_time = end;
}