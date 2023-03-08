#include "android_camera.h"
#include "backend.h"

namespace docscanner {
    struct pipeline {
        ACameraDevice* cam;
        uvec2 cam_out_size;
        ANativeWindow* texture_window;
        shader_program preview_program;
        variable proj_matrix_var;
        shader_buffer vbo;

        void pre_init_preview(int* cam_width, int* cam_height);
        void init_preview(uvec2 preview_size);
        void init_cam(ANativeWindow* texture_window);
        void render_preview();
    };
};