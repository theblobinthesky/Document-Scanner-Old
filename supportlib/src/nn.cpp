#include "nn.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"

using namespace docscanner;

TfLiteDelegate* create_delegate(execution_pref pref) {
    TfLiteNnapiDelegateOptions options = TfLiteNnapiDelegateOptionsDefault();
    
    if(pref == execution_pref::sustrained_speed)
        options.execution_preference = TfLiteNnapiDelegateOptions::ExecutionPreference::kSustainedSpeed;
    else if(pref == execution_pref::fast_single_answer)
        options.execution_preference = TfLiteNnapiDelegateOptions::ExecutionPreference::kFastSingleAnswer;
    
    return TfLiteNnapiDelegateCreate(&options);
}

void destroy_delegate(TfLiteDelegate* delegate) {
    TfLiteNnapiDelegateDelete(delegate);
}

#include "log.h"
#if false
void docscanner::create_neural_network_from_path(file_context* file_ctx, const char* path, execution_pref pref) {
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
    TfLiteTensorCopyFromBuffer(input_tensor, input.data(), input.size() * sizeof(float));

    TfLiteInterpreterInvoke(interpreter);

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    destroy_delegate(delegate);
}
#endif