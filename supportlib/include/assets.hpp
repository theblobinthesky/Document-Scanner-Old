#pragma once
#include "backend.hpp"
#include <vector>

struct TfLiteInterpreter;
struct TfLiteInterpreterOptions;
struct TfLiteModel;
struct TfLiteDelegate;
struct TfLiteTensor;

NAMESPACE_BEGIN

struct thread_pool;
struct engine_backend;

enum class execution_pref {
    sustained_speed,
    fast_single_answer
};

using texture_asset_id = u32;
using sdf_animation_asset_id = u32;
using font_asset_id = u32;
using nn_asset_id = u32;

enum class asset_state : u32 {
    queued, loaded
};

struct texture_asset {
    asset_state state;

    file_context* ctx;
    const char* path;
    texture tex;
};

struct sdf_animation_asset {
    asset_state state;
    
    // todo: implement this
};

struct font_asset {
    asset_state state;

    // todo: implement this
};

struct nn_asset {
    asset_state state;

    file_context* ctx;
    const char* path;
    TfLiteInterpreter* interpreter;
    TfLiteInterpreterOptions* options;
    TfLiteModel* model;
    TfLiteTensor* inp_ten;
};

// todo: deleteeeeee
#define MAX_ASSETS_PER_TYPE 24

struct asset_manager {
    file_context* ctx;
    thread_pool* threads;

    texture_asset texture_assets[MAX_ASSETS_PER_TYPE];
    u32 texture_assets_size;
    sdf_animation_asset sdf_animation_assets[MAX_ASSETS_PER_TYPE];
    u32 sdf_animation_assets_size;
    font_asset font_assets[MAX_ASSETS_PER_TYPE];
    u32 font_assets_size;
    nn_asset nn_assets[MAX_ASSETS_PER_TYPE];
    u32 nn_assets_size;
    
    asset_manager(file_context* ctx, thread_pool* thread);

    texture_asset_id load_texture_asset(const char* path);
    sdf_animation_asset_id load_sdf_animation_asset(const char* path);
    font_asset_id load_font_asset(const char* path);
    nn_asset_id load_nn_asset(const char* path);

    const texture_asset* get_texture_asset(texture_asset_id id);
    const sdf_animation_asset* get_sdf_animation_asset(sdf_animation_asset_id id);
    const font_asset* get_font_asset(font_asset_id id);
    const nn_asset* get_nn_asset(nn_asset_id id);
};

void destory_neural_network(asset_manager* assets, nn_asset_id id);
void invoke_neural_network_on_data(asset_manager* assets, nn_asset_id id, u8* inp_data, u32 inp_size, u8** out_datas, u32* out_sizes, u32 out_size);
NAMESPACE_END