#include "nn.hpp"
#include "log.hpp"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

using namespace docscanner;

#ifdef ANDROID
#if 0
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

neural_network docscanner::create_neural_network_from_path(file_context* file_ctx, const char* path, execution_pref pref) {
    TfLiteDelegate* delegate = create_delegate(pref);

    u8* model_data;
    u32 model_size;
    file_to_buffer(file_ctx, path, model_data, model_size);

    TfLiteModel* model = TfLiteModelCreate(model_data, model_size);
    
    auto options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
    
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    return {
        .interpreter=interpreter,
        .options = options,
        .model = model,
        .delegate = delegate,
        .inp_ten = input_tensor
    };
}

void docscanner::destory_neural_network(const neural_network& nn) {
    TfLiteInterpreterDelete(nn.interpreter);
    TfLiteInterpreterOptionsDelete(nn.options);
    TfLiteModelDelete(nn.model);
    destroy_delegate(nn.delegate);
}

#include <chrono>

void docscanner::invoke_neural_network_on_data(const neural_network& nn, u8* inp_data, u32 inp_size, u8** out_datas, u32* out_sizes, u32 out_size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    TfLiteTensorCopyFromBuffer(nn.inp_ten, inp_data, inp_size);
    TfLiteInterpreterInvoke(nn.interpreter);

    u32 got_out_size = TfLiteInterpreterGetOutputTensorCount(nn.interpreter);
    ASSERT(out_size == got_out_size, "Neural network has wrong amount of output tensors. Expected %u but got %u.", out_size, got_out_size);

    for(s32 i = 0; i < out_size; i++) {
        const TfLiteTensor* out_ten = TfLiteInterpreterGetOutputTensor(nn.interpreter, i);
        ASSERT(out_sizes[i] == out_ten->bytes, "Neural network output tensor size is wrong. Expected %u but got %u.", out_sizes[i], (u32)out_ten->bytes);

        TfLiteTensorCopyToBuffer(out_ten, out_datas[i], out_sizes[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOGI("Duration of nn inference: %lldms", dur.count());
}