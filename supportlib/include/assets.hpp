#pragma once
#include "utils.hpp"

#ifdef ANDROID
struct AAssetManager;
#endif

NAMESPACE_BEGIN

#ifdef ANDROID
struct file_context {
    AAssetManager* mngr;
    char* internal_data_path;
};

file_context get_file_ctx_from_asset_mngr(AAssetManager* mngr, char* internal_data_path);
#elif defined(LINUX)
struct file_context {};
#endif

void file_to_buffer(file_context* ctx, const char* path, u8* &data, u32 &size);

void read_from_internal_file(file_context* ctx, const char* path, u8* &data, u32 &size);
void write_to_internal_file(file_context* ctx, const char* path, u8* data, u32 size);

NAMESPACE_END