#include "types.hpp"
#include "backend.hpp"

#include <vector>

NAMESPACE_BEGIN

struct mask_mesher {
    shader_buffer mesh_buffer;
    shader_program mesh_program;
    std::vector<vertex> mesh_vertices;
    std::vector<u32> mesh_indices;

    f32* mask_buffer;
    svec2 mask_size;

    void init(f32* mask_buffer, const svec2& mask_size, float* projection);
    void mesh();
};

NAMESPACE_END