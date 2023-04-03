#include "utils.hpp"
#include "backend.hpp"
#include "assets.hpp"
#include "nn.hpp"
#include "img_proc.hpp"
#include "camera.hpp"
#include "shader_program.hpp"

#include <vector>

struct ACameraDevice;
struct ANativeWindow;

NAMESPACE_BEGIN

struct cam_preview {
    engine_backend* backend;

    shader_program preview_program;
    camera cam;

    shader_buffer cam_quad_buffer;

    u32 nn_input_buffer_size, nn_contour_out_size;
    f32 nn_exists_out;
    u8* nn_input_buffer, *nn_contour_out;

    neural_network nn;

    svec2 cam_tex_size;
    svec2 preview_size;

    u64 last_time;
    
    texture_downsampler tex_downsampler;
    mask_mesher mesher;
    sticky_particle_system particles;
    mesh_border border;
    mesh_cutout cutout;

    bool is_live_camera_streaming;
    animation<f32> unwrap_animation;
    animation<f32> blendout_animation, blendin_animation;

    bool is_init;

    cam_preview(engine_backend* backend);
    void pre_init(svec2 preview_size, int* cam_width, int* cam_height);
    void init_backend(file_context* file_ctx, f32 bottom_edge);

#ifdef ANDROID
    void init_cam(ANativeWindow* texture_window);
#elif defined(LINUX)
    void init_cam();
#endif

    void unwrap();
    void render(f32 time);
};

NAMESPACE_END