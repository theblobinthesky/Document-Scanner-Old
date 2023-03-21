#pragma once
#include "types.hpp"
#include "assets.hpp"

struct TfLiteInterpreter;
struct TfLiteInterpreterOptions;
struct TfLiteModel;
struct TfLiteDelegate;
struct TfLiteTensor;

NAMESPACE_BEGIN

enum class execution_pref {
    sustained_speed,
    fast_single_answer
};

struct neural_network {
    TfLiteInterpreter* interpreter;
    TfLiteInterpreterOptions* options;
    TfLiteModel* model;
    TfLiteDelegate* delegate;
    TfLiteTensor* inp_ten;
};

neural_network create_neural_network_from_path(file_context* file_ctx, const char* path, execution_pref pref);
void destory_neural_network(const neural_network& nn);
void invoke_neural_network_on_data(const neural_network& nn, u8* inp_data, u32 inp_size, u8** out_datas, u32* out_sizes, u32 out_size);

NAMESPACE_END