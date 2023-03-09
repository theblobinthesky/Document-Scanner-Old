#pragma once
#include "types.h"
#include "assets.h"

NAMESPACE_BEGIN

enum class execution_pref {
    sustrained_speed,
    fast_single_answer
};

void create_neural_network_from_path(file_context* file_ctx, const char* path, execution_pref pref);

NAMESPACE_END