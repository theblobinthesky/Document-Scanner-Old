#pragma once
#include "utils.hpp"

#ifdef ANDROID
struct AAssetManager;
#endif

struct TfLiteInterpreter;
struct TfLiteInterpreterOptions;
struct TfLiteModel;
struct TfLiteDelegate;
struct TfLiteTensor;

NAMESPACE_BEGIN

#ifdef ANDROID
struct file_context {
    AAssetManager* mngr;
    char* internal_data_path;
};

#elif defined(LINUX)
struct file_context {};
#endif

enum class execution_pref {
    sustained_speed,
    fast_single_answer
};

struct neural_network {
    bool was_initialized;
    TfLiteInterpreter* interpreter;
    TfLiteInterpreterOptions* options;
    TfLiteModel* model;
    TfLiteDelegate* delegate;
    TfLiteTensor* inp_ten;
};

struct asset_manager {
    file_context* ctx;
    neural_network* nn;

    void get_neural_network();
};

struct neural_network_params {
    asset_manager* assets;
    const char* path;
    execution_pref pref;
};

struct thread_pool;

#ifdef ANDROID
asset_manager* get_assets_from_asset_mngr(AAssetManager* mngr, char* internal_data_path);
#endif

void file_to_buffer(file_context* ctx, const char* path, u8* &data, u32 &size);

void read_from_internal_file(file_context* ctx, const char* path, u8* &data, u32 &size);
void write_to_internal_file(file_context* ctx, const char* path, u8* data, u32 size);

struct engine_backend;

void create_neural_network_from_path(asset_manager* assets, thread_pool* threads, const char* path, execution_pref pref);
// void destory_neural_network(const neural_network* nn);
void invoke_neural_network_on_data(asset_manager* assets, u8* inp_data, u32 inp_size, u8** out_datas, u32* out_sizes, u32 out_size);
NAMESPACE_END