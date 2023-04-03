#include "utils.hpp"
#include "backend.hpp"
#include "user_interface.hpp"
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
    ui_manager* ui;

    shader_program preview_program;
    camera cam;

    shader_buffer cam_quad_buffer;

    u32 nn_input_buffer_size, nn_contour_out_size;
    f32 nn_exists_out;
    u8* nn_input_buffer, *nn_contour_out;

    neural_network nn;

    texture_downsampler tex_downsampler;
    texture_sampler tex_sampler;

    mask_mesher mesher;
    sticky_particle_system particles;
    mesh_border border;
    mesh_cutout cutout;

    shader_program shutter_program;
    animation<f32> shutter_animation;

    bool is_live_camera_streaming;
    animation<f32> unwrap_animation;
    animation<f32> blendout_animation, blendin_animation;

    bool is_init;

    cam_preview(engine_backend* backend, ui_manager* ui, camera* cam);
    void init_backend(f32 bottom_edge, const rect& unwrapped_rect);

#ifdef ANDROID
    void init_cam(ANativeWindow* texture_window);
#elif defined(LINUX)
    void init_cam();
#endif

    void unwrap();
    void render(f32 time);
};

NAMESPACE_END