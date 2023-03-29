#include "cam_preview.hpp"
#include "log.hpp"
#include "backend.hpp"
#include "camera.hpp"
#include <chrono>

using namespace docscanner;

#ifdef ANDROID
#define CAM_USES_OES_TEXTURE true
#elif defined(LINUX)
#define CAM_USES_OES_TEXTURE false
#else
#error "Platform not supported yet."
#endif

void docscanner::cam_preview::pre_init(uvec2 preview_size, int* cam_width, int* cam_height) {
    this->preview_size = preview_size;

    cam = find_and_open_back_camera(preview_size, cam_tex_size);
    *cam_width = (int) cam_tex_size.x;
    *cam_height = (int) cam_tex_size.y;
}

void docscanner::cam_preview::init_backend(file_context* file_ctx) {
    float p = (cam_tex_size.x / (float) cam_tex_size.y) * (preview_size.x / (float) preview_size.y);
    f32 cam_tex_left = (1.0f - p) / 2.0f;
    f32 cam_tex_right = 1.0f - cam_tex_left;
    
    mat4 projection;
    mat4f_load_ortho(cam_tex_left, cam_tex_right, 0.0f, 1.0f, -1.0f, 1.0f, projection.data);
    backend.init(projection);

    preview_program = backend.compile_and_link(vert_src, frag_simple_tex_sampler_src(CAM_USES_OES_TEXTURE, 0));

    // buffer stuff
    vertex vertices[] = {
        {{1.f, 0.f}, {1, 0}},
        {{0.f, 1.f}, {0, 1}},
        {{1.f, 1.f}, {0, 0}},
        {{0.f, 0.f}, {1, 1}}
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
    tex_downsampler.init(&backend, cam_tex_size, downsampled_size, true, null, 1.0);
#else
    tex_downsampler.init(&backend, cam_tex_size, downsampled_size, false, &cam.cam_tex, 2.0);
#endif

    mesher.init(&nn_exists_out, (f32*)nn_contour_out, downsampled_size, 0.2f);

    auto buffer = make_shader_buffer();

    particles.init(&backend, mesher.vertices, mesher.mesh_size, buffer);
    border.init(&backend, mesher.left_contour, mesher.top_contour, mesher.right_contour, mesher.bottom_contour, mesher.mesh_size.x, buffer);
    
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

void docscanner::cam_preview::render(f32 time) {
    if(!is_init) return;
    cam.get();

    nn_input_tex = tex_downsampler.downsample();

    get_framebuffer_data(tex_downsampler.output_fb, tex_downsampler.output_size, nn_input_buffer, nn_input_buffer_size);
    
    constexpr u32 out_size = 1; // 2;
    u8* out_datas[out_size] = { nn_contour_out }; //, (u8*)&nn_exists_out };
    u32 out_sizes[out_size] = { nn_contour_out_size }; // , sizeof(f32) };
    invoke_neural_network_on_data(nn, nn_input_buffer, nn_input_buffer_size, out_datas, out_sizes, out_size);

    mesher.mesh(&backend);

    canvas c = {
        .bg_color={0, 1, 0}
    };
    
    use_program(preview_program);

    unbind_framebuffer();
    bind_shader_buffer(cam_quad_buffer);
    
#if !CAM_USES_OES_TEXTURE
    bind_texture_to_slot(0, cam.cam_tex);
#endif

    draw(c);

#if true
    if(mesher.does_mesh_exist()) {
        particles.render();
        border.render(time);
    }
#endif

    backend.DEBUG_draw();

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    auto dur = end - last_time;
    LOGI("frame time: %ums", (u32)dur);
    last_time = end;
}