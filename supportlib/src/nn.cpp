#include "nn.h"
#include "log.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

using namespace docscanner;

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
#elif false
TfLiteDelegate* create_delegate(execution_pref pref) {
    auto options = TfLiteGpuDelegateOptionsV2Default();

    if(pref == execution_pref::sustained_speed)
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    else if(pref == execution_pref::fast_single_answer)
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;

    return TfLiteGpuDelegateV2Create(&options);
}
#else
TfLiteDelegate* create_delegate(execution_pref pref) {
    auto options = TfLiteXNNPackDelegateOptionsDefault();

    return TfLiteXNNPackDelegateCreate(&options);
}
#endif

void destroy_delegate(TfLiteDelegate* delegate) {
    TfLiteNnapiDelegateDelete(delegate);
}

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

    return {
        .interpreter=interpreter,
        .options = options,
        .model = model,
        .delegate = delegate
    };
}

void docscanner::destory_neural_network(const neural_network& nn) {
    TfLiteInterpreterDelete(nn.interpreter);
    TfLiteInterpreterOptionsDelete(nn.options);
    TfLiteModelDelete(nn.model);
    destroy_delegate(nn.delegate);
}

#include <chrono>

void docscanner::invoke_neural_network_on_data(const neural_network& nn, u8* inp_data, u32 inp_size, u8* out_data, u32 out_size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(nn.interpreter, 0);
    TfLiteTensorCopyFromBuffer(input_tensor, inp_data, inp_size);

    TfLiteInterpreterInvoke(nn.interpreter);

    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(nn.interpreter, 0);
    TfLiteTensorCopyToBuffer(output_tensor, out_data, out_size);

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOGI("duration of nn inference: %lldms", dur.count());
}