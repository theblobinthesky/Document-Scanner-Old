#pragma once
#include "types.hpp"

struct ACameraDevice;
struct ANativeWindow;

NAMESPACE_BEGIN

ACameraDevice* find_and_open_back_camera(const uvec2& min_size, uvec2& size);

void init_camera_capture_to_native_window(ACameraDevice* cam, ANativeWindow* texture_window);

NAMESPACE_END