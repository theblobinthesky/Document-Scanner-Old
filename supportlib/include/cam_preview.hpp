#include "types.hpp"
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
    shader_programmer programmer;

    shader_program preview_program;
    camera cam;

    shader_buffer cam_quad_buffer;

    u32 nn_input_buffer_size, nn_mask_out_size, nn_flatten_out_size;
    u8* nn_input_buffer, *nn_mask_out_buff, *nn_flatten_out_buff;

    texture* nn_input_tex;
    texture nn_output_tex;

    neural_network nn;

    uvec2 cam_tex_size;
    uvec2 preview_size;
    f32 cam_tex_left, cam_tex_right;

    u64 last_time;
    
    texture_downsampler tex_downsampler;
    mask_mesher mesher;
    sticky_particle_system particles;
    mesh_border border;

    bool is_init;

    void pre_init(uvec2 preview_size, int* cam_width, int* cam_height);
    void init_backend(file_context* file_ctx);

#ifdef ANDROID
    void init_cam(ANativeWindow* texture_window);
#elif defined(LINUX)
    void init_cam();
#endif

    void render();
};

NAMESPACE_END