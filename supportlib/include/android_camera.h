#pragma once
#include "types.h"

struct ACameraDevice;
struct ANativeWindow;

ACameraDevice* find_and_open_back_camera(u32 &width, u32 &height);

void init_camera_capture_to_native_window(ACameraDevice* cam, ANativeWindow* texture_window);