#pragma once
#include "types.hpp"

#ifdef ANDROID
struct ACameraDevice;
struct ANativeWindow;
#endif 

#ifdef LINUX
#endif

NAMESPACE_BEGIN

#ifdef ANDROID
struct camera {
    ACameraDevice* device;
};
#elif defined(LINUX)
struct camera {
    u8* buffer;
};
#endif

camera find_and_open_back_camera(const uvec2& min_size, uvec2& size);

#ifdef ANDROID
void init_camera_capture_to_native_window(const camera& cam, ANativeWindow* texture_window);
#elif defined(LINUX)

#endif

NAMESPACE_END