#include "assets.hpp"
#include "backend.hpp"
#include "log.hpp"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

using namespace docscanner;

#ifdef ANDROID
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <string>

std::string get_internal_path(file_context* ctx, const char* path) {
    return std::string(ctx->internal_data_path) + "/" + std::string(path);
}

#ifdef ANDROID
asset_manager* docscanner::get_assets_from_asset_mngr(AAssetManager* mngr, char* internal_data_path) {
    asset_manager* assets = new asset_manager();
    
    file_context* ctx = new file_context();
    ctx->mngr = mngr;
    ctx->internal_data_path = internal_data_path;
    assets->ctx = ctx;

    return assets;
}
#endif

void docscanner::file_to_buffer(file_context* ctx, const char* path, u8* &data, u32 &size) {
    AAsset* asset = AAssetManager_open(ctx->mngr, path, AASSET_MODE_BUFFER);
    ASSERT(asset != null, "AAsset open failed.");
    
    size = AAsset_getLength(asset);
    data = new u8[size];

    int status = AAsset_read(asset, data, size);
    ASSERT(status >= 0, "AAsset read failed.");
}

void docscanner::read_from_internal_file(file_context* ctx, const char* path, u8* &data, u32 &size) {
    std::string full_path = get_internal_path(ctx, path);

    FILE* file = fopen(full_path.c_str(), "rb");
    if(!file) return;

    fseek(file, 0, SEEK_END);
    size = (u32)ftell(file);
    rewind(file);

    data = new u8[size];
    fread(data, size, 1, file);
    fclose(file);
}

void docscanner::write_to_internal_file(file_context* ctx, const char* path, u8* data, u32 size) {
    std::string full_path = get_internal_path(ctx, path);
    
    FILE* file = fopen(full_path.c_str(), "wb");
    fwrite(data, size, 1, file);
    fclose(file);
}

#elif defined(LINUX)
#include <stdio.h>

void docscanner::file_to_buffer(file_context* ctx, const char* path, u8* &data, u32 &size) {
    FILE* file = fopen(path, "rb");
    if(!file) {
        LOGE("Path '%s' does not exist.", path);
        return;
    }

    fseek(file, 0, SEEK_END);
    size = (u32)ftell(file);
    rewind(file);

    data = new u8[size];
    fread(data, 1, size, file);

    fclose(file);
}

#endif

#ifdef ANDROID
#if true
TfLiteDelegate* create_delegate(execution_pref pref) {
    TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();

    if(pref == execution_pref::sustained_speed)
        options.execution_preference = TfLiteNnapiDelegateOptions::ExecutionPreference::kSustainedSpeed;
    else if(pref == execution_pref::fast_single_answer)
        options.execution_preference = TfLiteNnapiDelegateOptions::ExecutionPreference::kFastSingleAnswer;

    options.allow_fp16 = true;

    return TfLiteNnapiDelegateCreate(&options);
}

void destroy_delegate(TfLiteDelegate* delegate) {
    TfLiteNnapiDelegateDelete(delegate);
}
#else
TfLiteDelegate* create_delegate(execution_pref pref) {
    auto options = TfLiteGpuDelegateOptionsV2Default();

    if(pref == execution_pref::sustained_speed)
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    else if(pref == execution_pref::fast_single_answer)
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;

    return TfLiteGpuDelegateV2Create(&options);
}

void destroy_delegate(TfLiteDelegate* delegate) {
    TfLiteGpuDelegateV2Delete(delegate);
}
#endif
#elif defined(LINUX)
TfLiteDelegate* create_delegate(execution_pref pref) {
    auto options = TfLiteXNNPackDelegateOptionsDefault();

    return TfLiteXNNPackDelegateCreate(&options);
}


void destroy_delegate(TfLiteDelegate* delegate) {
    TfLiteXNNPackDelegateDelete(delegate);
}
#endif

void create_neural_network_internal(void* data) {
    neural_network_params* params = reinterpret_cast<neural_network_params*>(data);
    neural_network* nn = new neural_network();

    TfLiteDelegate* delegate = create_delegate(params->pref);

    u8* model_data;
    u32 model_size;
    file_to_buffer(params->assets->ctx, params->path, model_data, model_size);

    TfLiteModel* model = TfLiteModelCreate(model_data, model_size);
    
    auto options = TfLiteInterpreterOptionsCreate();
    // TfLiteInterpreterOptionsAddDelegate(options, delegate);
    
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    nn->interpreter = interpreter;
    nn->options = options;
    nn->model = model;
    nn->delegate = delegate;
    nn->inp_ten = input_tensor;
    nn->was_initialized = true;

    params->assets->nn = nn;
}

void docscanner::create_neural_network_from_path(asset_manager* assets, thread_pool* threads, const char* path, execution_pref pref) {
    neural_network_params* params = new neural_network_params({ assets, path, pref });
    threads->push({ create_neural_network_internal, params });
}

/*
void docscanner::destory_neural_network(const neural_network* nn) {
    TfLiteInterpreterDelete(nn->interpreter);
    TfLiteInterpreterOptionsDelete(nn->options);
    TfLiteModelDelete(nn->model);
    destroy_delegate(nn->delegate);
}
*/

void docscanner::invoke_neural_network_on_data(asset_manager* assets, u8* inp_data, u32 inp_size, u8** out_datas, u32* out_sizes, u32 out_size) {
    auto nn = assets->nn;
    if(!nn || !nn->was_initialized) return; // todo: this is bad code! fix it later.

    ASSERT(inp_size == nn->inp_ten->bytes, "Neural network input tensor size is wrong. Expected %u but got %u.", inp_size, (u32)nn->inp_ten->bytes);

    TfLiteTensorCopyFromBuffer(nn->inp_ten, inp_data, inp_size);
    TfLiteInterpreterInvoke(nn->interpreter);

    u32 got_out_size = TfLiteInterpreterGetOutputTensorCount(nn->interpreter);
    ASSERT(out_size == got_out_size, "Neural network has wrong amount of output tensors. Expected %u but got %u.", out_size, got_out_size);

    for(s32 i = 0; i < out_size; i++) {
        const TfLiteTensor* out_ten = TfLiteInterpreterGetOutputTensor(nn->interpreter, i);
        ASSERT(out_sizes[i] == out_ten->bytes, "Neural network output tensor size is wrong. Expected %u but got %u.", out_sizes[i], (u32)out_ten->bytes);

        TfLiteTensorCopyToBuffer(out_ten, out_datas[i], out_sizes[i]);
    }
}