#include "assets.h"
#ifdef ANDROID
#include "log.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

using namespace docscanner;

file_context docscanner::get_file_ctx_from_asset_mngr(AAssetManager* mngr) {
    return {mngr};
}

void docscanner::file_to_buffer(file_context* ctx, const char* path, u8* &data, u32 &size) {
    AAsset* asset = AAssetManager_open(ctx->mngr, path, AASSET_MODE_BUFFER);
    ASSERT(asset != null, "AAsset open failed.");
    
    size = AAsset_getLength(asset);
    data = new u8[size];

    int status = AAsset_read(asset, data, size);
    ASSERT(status >= 0, "AAsset read failed.");
}

#endif