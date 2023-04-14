#include "assets.hpp"
#include "log.hpp"

using namespace docscanner;

#ifdef ANDROID
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <string>

std::string get_internal_path(file_context* ctx, const char* path) {
    return std::string(ctx->internal_data_path) + "/" + std::string(path);
}

file_context docscanner::get_file_ctx_from_asset_mngr(AAssetManager* mngr, char* internal_data_path) {
    return { mngr, internal_data_path };
}

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