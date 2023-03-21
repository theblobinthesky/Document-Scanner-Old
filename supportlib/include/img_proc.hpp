#include "types.hpp"
#include "backend.hpp"

#include <vector>

NAMESPACE_BEGIN

struct mask_mesher {
    shader_buffer mesh_buffer;
    std::vector<vertex> mesh_vertices;
    std::vector<u32> mesh_indices;
    svec2 mesh_size;

    f32* mask_buffer, *flatten_buffer;
    svec2 mask_size;

    bool exists;

    void init(shader_programmer* programmer, f32* mask_buffer, f32* flatten_buffer, const svec2& mask_size, float* projection);
    void mesh();
};

NAMESPACE_END