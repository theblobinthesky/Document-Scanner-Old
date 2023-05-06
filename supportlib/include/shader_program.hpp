#pragma once
#include "utils.hpp"
#include <unordered_map>
#include <string>

NAMESPACE_BEGIN
struct shader_program;

std::string vert_src();

std::string vert_quad_src();

std::string vert_instanced_quad_src();

std::string vert_instanced_point_src();

std::string vert_instanced_line_src();

std::string frag_glyph_src(u32 binding_slot);

std::string frag_gauss_blur_src(bool use_oes, u32 N, const vec2& pixel_shift);

std::string frag_debug_src();

std::string frag_sampler_src(bool use_oes);

std::string frag_DEBUG_marker_src();

std::string frag_border_src();

std::string frag_particle_src();

std::string frag_shutter_src();

std::string frag_rounded_colored_quad_desc_src();

std::string frag_rounded_textured_quad_desc_src(bool use_oes);

std::string frag_sdf_quad_desc_src();

NAMESPACE_END