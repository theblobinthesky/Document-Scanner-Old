#include "utils.hpp"
#include "backend.hpp"
#include "user_interface.hpp"
#include "assets.hpp"
#include "img_proc.hpp"
#include "shader_program.hpp"

#include <vector>

struct ACameraDevice;
struct ANativeWindow;

NAMESPACE_BEGIN

struct cam_preview {
    engine_backend* backend;
    ui_manager* ui;

    rect cam_pos_bounds;
    rect cam_uv_bounds;

    nn_asset_id nn_id;
    u32 nn_input_buffer_size, nn_contour_out_size;
    f32 nn_exists_out;
    u8* nn_input_buffer, *nn_contour_out;

    texture_downsampler tex_downsampler;
    texture_sampler tex_sampler;
    vertex* unwrapped_vertices;

    mask_mesher mesher;
    sticky_particle_system particles;
    mesh_border border;
    mesh_cutout cutout;

    sdf_button flash_button;
    shader_program shutter_program;
    animation<f32> shutter_animation;

    animation<f32> unwrap_animation, blendout_animation;
    
    bool is_live_camera_streaming;
    bool is_init, is_visible;

    f32 bottom_edge; // hack
    rect unwrapped_rect;
    cam_preview(engine_backend* backend, ui_manager* ui, f32 bottom_edge, const rect& unwrapped_rect);
    void init_camera_related();

    void unwrap();
    void draw_ui();
    void draw_unwrapped_ui();
    void render();
};

NAMESPACE_END