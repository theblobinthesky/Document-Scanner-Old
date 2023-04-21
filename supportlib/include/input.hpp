#pragma once
#include "utils.hpp"

NAMESPACE_BEGIN

enum class motion_type : s32 {
    NO_MOTION,
    TOUCH_DOWN, TOUCH_UP,
    MOVE
};

struct motion_event {
    motion_type type;
    vec2 pos;
};

struct input_manager {
    motion_event event;
    svec2 preview_size;
    f32 aspect_ratio;

    void init(svec2 preview_size, f32 aspect_ratio);
    void handle_motion_event(const motion_event& event);
    motion_event get_motion_event(const rect& bounds);
    void end_frame();
};

NAMESPACE_END