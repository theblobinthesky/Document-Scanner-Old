#include "types.hpp"
#include "backend.hpp"
#include "assets.hpp"
#include "nn.hpp"
#include "img_proc.hpp"
#include "camera.hpp"

#include <vector>

struct ACameraDevice;
struct ANativeWindow;

NAMESPACE_BEGIN

struct cam_preview {
    shader_program preview_program;
    camera cam;

    shader_buffer cam_quad_buffer;

    u32 nn_input_buffer_size, nn_output_buffer_size;
    u8* nn_input_buffer, *nn_output_buffer;

    texture* nn_input_tex;
    texture nn_output_tex;

    neural_network nn;

    uvec2 cam_tex_size;
    uvec2 preview_size;
    f32 cam_tex_left, cam_tex_right;

    u64 last_time;
    
    texture_downsampler tex_downsampler;
    mask_mesher mesher;

    shader_program border_program;
    std::vector<s32> border_indices;
    std::vector<vertex> border_vertices;

    bool is_init;

    void pre_init(uvec2 preview_size, int* cam_width, int* cam_height);
    void init_backend(file_context* file_ctx);
    void init_cam(ANativeWindow* texture_window);
    void render();
};

NAMESPACE_END