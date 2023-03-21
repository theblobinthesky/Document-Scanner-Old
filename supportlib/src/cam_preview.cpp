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

constexpr u32 flatten_f32_count = 9;

void docscanner::cam_preview::pre_init(uvec2 preview_size, int* cam_width, int* cam_height) {
    this->preview_size = preview_size;

    cam = find_and_open_back_camera(preview_size, cam_tex_size);
    *cam_width = (int) cam_tex_size.x;
    *cam_height = (int) cam_tex_size.y;
}

void docscanner::cam_preview::init_backend(file_context* file_ctx) {
    glEnable(GL_BLEND);  
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

    preview_program = programmer.compile_and_link(vert_src, frag_simple_tex_sampler_src(CAM_USES_OES_TEXTURE, 0));

    // buffer stuff
    float p = (cam_tex_size.x / (float) cam_tex_size.y) * (preview_size.x / (float) preview_size.y);
    cam_tex_left = (1.0f - p) / 2.0f;
    cam_tex_right = 1.0f - cam_tex_left;
    
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

    use_program(preview_program);

    auto proj_matrix_var = get_variable(preview_program, "projection");
    
    float projection[16];
    mat4f_load_ortho(cam_tex_left, cam_tex_right, 0.0f, 1.0f, -1.0f, 1.0f, projection);

    proj_matrix_var.set_mat4(projection);

    cam_quad_buffer = make_shader_buffer();
    fill_shader_buffer(cam_quad_buffer, vertices, sizeof(vertices), indices, sizeof(indices));

    uvec2 downsampled_size = {64, 64};

    nn_input_buffer_size = downsampled_size.x * downsampled_size.y * 4 * sizeof(float);
    nn_input_buffer = new u8[nn_input_buffer_size];

    nn_flatten_out_size = flatten_f32_count * sizeof(f32);
    nn_mask_out_size = downsampled_size.x * downsampled_size.y * 1 * sizeof(float);
    nn_flatten_out_buff = new u8[flatten_f32_count * sizeof(f32)];
    nn_mask_out_buff = new u8[nn_mask_out_size];
    
    nn_output_tex = create_texture(downsampled_size , GL_R32F);
    glBindTexture(GL_TEXTURE_2D, nn_output_tex.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#if CAM_USES_OES_TEXTURE
    tex_downsampler.init(&programmer, cam_tex_size, downsampled_size, true, null, 2.0);
#else
    tex_downsampler.init(&programmer, cam_tex_size, downsampled_size, false, &cam.cam_tex, 2.0);
#endif

    mesher.init(&programmer, (f32*)nn_mask_out_buff, (f32*)nn_flatten_out_buff, { (s32)downsampled_size.x, (s32)downsampled_size.y }, projection);

    auto buffer = make_shader_buffer();

    particles.init(&programmer, mesher.mesh_vertices, mesher.mesh_size, projection, buffer);
    border.init(&programmer, mesher.mesh_vertices, mesher.mesh_size, projection, buffer);
    
    nn = create_neural_network_from_path(file_ctx, "seg_model.tflite", execution_pref::sustained_speed);

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
    
    constexpr u32 out_size = 2;
    u8* out_datas[out_size] = { nn_flatten_out_buff, nn_mask_out_buff };
    u32 out_sizes[out_size] = { nn_flatten_out_size, nn_mask_out_size };
    invoke_neural_network_on_data(nn, nn_input_buffer, nn_input_buffer_size, out_datas, out_sizes, out_size);

    mesher.mesh();

    set_texture_data(nn_output_tex, nn_mask_out_buff, tex_downsampler.output_size.x, tex_downsampler.output_size.y);

    canvas c = {
        .bg_color={0, 1, 0}
    };
    
    use_program(preview_program);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(cam_quad_buffer.id);
    
#if !CAM_USES_OES_TEXTURE
    bind_texture_to_slot(0, cam.cam_tex);
#endif

    draw(c);

    if(mesher.exists) {
        particles.render();
        border.render(time);
    }

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    auto dur = end - last_time;
    LOGI("frame time: %ums", (u32)dur);
    last_time = end;
    return;
}