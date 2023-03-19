#pragma once
#include "types.hpp"

#ifdef ANDROID
struct AAssetManager;
#endif

NAMESPACE_BEGIN

#ifdef ANDROID
struct file_context {
    AAssetManager* mngr;
};

file_context get_file_ctx_from_asset_mngr(AAssetManager* mngr);
#elif defined(LINUX)
struct file_context {};
#endif

void file_to_buffer(file_context* ctx, const char* path, u8* &data, u32 &size);

NAMESPACE_END