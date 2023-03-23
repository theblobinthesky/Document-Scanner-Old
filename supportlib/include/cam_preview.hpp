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

extern const s32 points_per_side_incl_start_corner;

struct cam_preview {
    engine_backend backend;

    shader_program preview_program;
    camera cam;

    shader_buffer cam_quad_buffer;

    u32 nn_input_buffer_size, nn_contour_out_size;
    f32 nn_exists_out;
    u8* nn_input_buffer, *nn_contour_out;

    texture* nn_input_tex;

    neural_network nn;

    uvec2 cam_tex_size;
    uvec2 preview_size;

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

    void render(f32 time);
};

NAMESPACE_END