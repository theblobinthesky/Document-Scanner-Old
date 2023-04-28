#include "assets.hpp"
#include "backend.hpp"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "stb_image.h"
#include <string.h>

using namespace docscanner;

enum class asset_type : u32 {
    image, sdf_animation, font, neural_network
};

struct data_internal {
    thread_pool* threads;
    void* data;
};

f32* read_texture_from_data(u8* data, u32 data_size) {
    svec2 size;
    s32 channels;
    unsigned char *stbi_data = stbi_load_from_memory(data, data_size, &size.x, &size.y, &channels, 3);

    f32* cvt_f32 = new f32[size.area() * 4];
    memset(cvt_f32, 0, sizeof(f32) * size.area() * 4);

    if(channels == 3) {
    
        for(s32 i = 0; i < size.area(); i++) {
            s32 idx = i * 4;
            
            for(s32 c = 0; c < 3; c++) {
                cvt_f32[idx + c] = stbi_data[i * 3 + c] / 255.0f;
            }

            cvt_f32[idx + 3] = 1.0f;
        }
    
    } else if(channels == 4) {
        
        for(s32 i = 0; i < size.area(); i++) {
            s32 idx = i * 4;
            
            for(s32 c = 0; c < 4; c++) {
                cvt_f32[idx + c] = stbi_data[i * 3 + c] / 255.0f;
            }
        }

    } else {
        LOGE_AND_BREAK("Something other than 3/4 channels is not supported right now.");
    }

    stbi_image_free(stbi_data);
    return cvt_f32;
}

void upload_texture_asset_internal(void* data) {
    texture_asset* asset = reinterpret_cast<texture_asset*>(data);

    asset->tex = make_texture(asset->image_size, GL_RGBA32F);
    set_texture_data(asset->tex, (u8*)asset->image_data, asset->image_size);

    asset->state = asset_state::loaded;
}

void load_texture_asset_internal(void* data) {
    data_internal* internal = reinterpret_cast<data_internal*>(data);
    texture_asset* asset = reinterpret_cast<texture_asset*>(internal->data);

    asset->image_data = read_texture_from_data(asset->data, asset->data_size);
    internal->threads->push_gui({ upload_texture_asset_internal, asset });
}

void load_nn_asset_internal(void* data) {
    nn_asset* asset = reinterpret_cast<nn_asset*>(data);

    // todo: fix the delegate "situation" TfLiteDelegate* delegate = create_gpu_delegate(execution_pref::sustained_speed);

    TfLiteModel* model = TfLiteModelCreate(asset->data, asset->data_size);
    
    auto options = TfLiteInterpreterOptionsCreate();
    // TfLiteInterpreterOptionsAddDelegate(options, delegate);
    
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    asset->state = asset_state::loaded;
    asset->interpreter = interpreter;
    asset->options = options;
    asset->model = model;
    asset->inp_ten = input_tensor;
}

void asset_manager::parse_package(u8* data, u32 data_size) {
    u32 table_of_contents_offset = *reinterpret_cast<u32*>(data + data_size - 8);
    u32 entries_size = *reinterpret_cast<u32*>(data + data_size - 4);

    LOGI("asset_manager table_of_contents_offset: %u", table_of_contents_offset);
    LOGI("asset_manager entries_size: %u", entries_size);

    u32* entries_offsets = reinterpret_cast<u32*>(data + table_of_contents_offset);
    for(u32 i = 0; i < entries_size; i++) {
        u32* entry_ptr = reinterpret_cast<u32*>(data + entries_offsets[i]);
        
        u32 entry_type = entry_ptr[0];
        u32 name_offset = entry_ptr[1];
        u32 name_size = entry_ptr[2];
        u32 entry_data_offset = entry_ptr[3];
        u32 entry_data_size = entry_ptr[4];

        char* name = (char*)(data + name_offset);
        u8* entry_data = data + entry_data_offset;
        // LOGI("asset_manager entry_type: %u, name: %s, name_size: %u, entry_data_offset: %u, entry_data_size: %u", entry_type, name, name_size, entry_data_offset, entry_data_size);

        switch(entry_type) {
            case static_cast<u32>(asset_type::image): {
                texture_asset asset = {};
                
                asset.state = asset_state::waiting;
                asset.path = name;
                asset.path_size = name_size;
                asset.data = entry_data;
                asset.data_size = entry_data_size;

                asset.image_size = { (s32)entry_ptr[5], (s32)entry_ptr[6] };
                asset.image_channels = entry_ptr[7];

                texture_assets[texture_assets_size++] = asset;
            } break;
            case static_cast<u32>(asset_type::font): {
                font_asset asset = {};

                asset.state = asset_state::waiting;
                asset.path = name;
                asset.path_size = name_size;
                asset.data = entry_data;
                asset.data_size = entry_data_size;

                font_assets[font_assets_size++] = asset;
            } break;
            case static_cast<u32>(asset_type::neural_network): {
                nn_asset asset = {};

                asset.state = asset_state::waiting;
                asset.path = name;
                asset.path_size = name_size;
                asset.data = entry_data;
                asset.data_size = entry_data_size;

                nn_assets[nn_assets_size++] = asset;
            } break;
        }
    }
}

asset_manager::asset_manager(file_context* ctx, const char* package_file, thread_pool* threads) : ctx(ctx), threads(threads) {
    SCOPED_TIMER("ASSET MANAGER SYNCHRONOUS OVERHEAD");
    texture_assets_size = 0;
    sdf_animation_assets_size = 0;
    font_assets_size = 0;
    nn_assets_size = 0;

    u8* data;
    u32 data_size;
    read_from_package(ctx, package_file, data, data_size);
    parse_package(data, data_size);
}

texture_asset_id asset_manager::load_texture_asset(const char* path) {
    texture_asset_id id = -1;

    for(s32 i = 0; i < texture_assets_size; i++) {
        const texture_asset& asset = texture_assets[i];        
        if(memcmp(asset.path, path, asset.path_size) == 0) {
            id = i;
            break;
        }
    }

    ASSERT(id != -1, "Texture asset name '%s' does not exist.", path);

    if(texture_assets[id].state == asset_state::waiting) {
        // @Memory leak
        data_internal* internal = new data_internal({ threads, &texture_assets[id] });
        threads->push({ load_texture_asset_internal, internal });
    }

    return id;
}

sdf_animation_asset_id asset_manager::load_sdf_animation_asset(const char* path) {
    LOGE_AND_BREAK("not implemented.");
    return 0;
}

font_asset_id asset_manager::load_font_asset(const char* path) {
    font_asset_id id = -1;

    for(s32 i = 0; i < font_assets_size; i++) {
        const font_asset& asset = font_assets[i];        
        if(memcmp(asset.path, path, asset.path_size) == 0) {
            id = i;
            break;
        }
    }

    ASSERT(id != -1, "Font asset name '%s' does not exist.", path);

    if(font_assets[id].state == asset_state::waiting) {
        // todo: fix this threads->push({ load_font_asset_internal, &font_assets[id] });
    }

    return id;
}

nn_asset_id asset_manager::load_nn_asset(const char* path) {
    nn_asset_id id = -1;

    for(s32 i = 0; i < nn_assets_size; i++) {
        const nn_asset& asset = nn_assets[i];        
        if(memcmp(asset.path, path, asset.path_size) == 0) {
            id = i;
            break;
        }
    }

    ASSERT(id != -1, "Neural network asset name '%s' does not exist.", path);

    if(nn_assets[id].state == asset_state::waiting) {
        threads->push({ load_nn_asset_internal, &nn_assets[id] });
    }

    return id;
}

const texture_asset* asset_manager::get_texture_asset(texture_asset_id id) {
    return &texture_assets[id];
}

const sdf_animation_asset* asset_manager::get_sdf_animation_asset(sdf_animation_asset_id id) {
    return &sdf_animation_assets[id];
}

const font_asset* asset_manager::get_font_asset(font_asset_id id) {
    return &font_assets[id];
}

const nn_asset* asset_manager::get_nn_asset(nn_asset_id id) {
    return &nn_assets[id];
}

#if false
#if false
#elif defined(LINUX)
#include <stdio.h>

void docscanner::read_from_package(file_context* ctx, const char* path, u8* &data, u32 &size) {
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
#endif

#ifdef false
TfLiteDelegate* create_nnapi_delegate(execution_pref pref) {
    TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();

    if(pref == execution_pref::sustained_speed)
        options.execution_preference = TfLiteNnapiDelegateOptions::ExecutionPreference::kSustainedSpeed;
    else if(pref == execution_pref::fast_single_answer)
        options.execution_preference = TfLiteNnapiDelegateOptions::ExecutionPreference::kFastSingleAnswer;

    options.allow_fp16 = true;

    return TfLiteNnapiDelegateCreate(&options);
}

TfLiteDelegate* create_xnnpack_delegate(execution_pref pref) {
    auto options = TfLiteXNNPackDelegateOptionsDefault();

    return TfLiteXNNPackDelegateCreate(&options);
}

TfLiteDelegate* create_gpu_delegate(execution_pref pref) {
    auto options = TfLiteGpuDelegateOptionsV2Default();

    if(pref == execution_pref::sustained_speed)
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    else if(pref == execution_pref::fast_single_answer)
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;

    return TfLiteGpuDelegateV2Create(&options);
}

void destroy_nnapi_delegate(TfLiteDelegate* delegate) {
    TfLiteNnapiDelegateDelete(delegate);
}

void destroy_xnnpack_delegate(TfLiteDelegate* delegate) {
    TfLiteXNNPackDelegateDelete(delegate);
}

void destroy_gpu_delegate(TfLiteDelegate* delegate) {
    TfLiteGpuDelegateV2Delete(delegate);
}
#endif

void docscanner::destory_neural_network(asset_manager* assets, nn_asset_id id) {
    const nn_asset* asset = assets->get_nn_asset(id);

    TfLiteInterpreterDelete(asset->interpreter);
    TfLiteInterpreterOptionsDelete(asset->options);
    TfLiteModelDelete(asset->model);
}

void docscanner::invoke_neural_network_on_data(asset_manager* assets, nn_asset_id id, u8* inp_data, u32 inp_size, u8** out_datas, u32* out_sizes, u32 out_size) {
    auto nn = assets->get_nn_asset(id);
    if(!nn || nn->state != asset_state::loaded) return;

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