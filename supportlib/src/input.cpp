#include "input.hpp"
#include "log.hpp"

using namespace docscanner;

void input_manager::init(svec2 preview_size, f32 aspect_ratio) {
    this->preview_size = preview_size;
    this->aspect_ratio = aspect_ratio;
}

void docscanner::input_manager::handle_motion_event(const motion_event& event) {
    this->event = {
        .type = event.type,
        .pos = { event.pos.x / (f32)(preview_size.x - 1), aspect_ratio * event.pos.y / (f32)(preview_size.y - 1) }
    };
    LOGI("event.pos.x: %f, event.pos.y: %f", this->event.pos.x, this->event.pos.y);
}

motion_event input_manager::get_motion_event(const vec2& tl, const vec2& br) {
    if(event.type != motion_type::NO_MOTION &&
        tl.x <= event.pos.x && event.pos.x <= br.x &&
        tl.y <= event.pos.y && event.pos.y <= br.y) {
        return event;
    }

    return {
        .type = motion_type::NO_MOTION,
        .pos = {}
    };
}

void input_manager::end_frame() {
    event = {
        .type = motion_type::NO_MOTION,
        .pos = {}
    };
}