#include "platform.hpp"

using namespace docscanner;

void input_manager::init(svec2 preview_size, f32 aspect_ratio) {
    this->preview_size = preview_size;
    this->aspect_ratio = aspect_ratio;
}

void docscanner::input_manager::handle_motion_event(const motion_event& event) {
    LOGI("MOTION_EVENT: %d, %f, %f", (s32)event.type, event.pos.x, event.pos.y);
    
    this->event = {
        .type = event.type,
        .pos = { event.pos.x / (f32)(preview_size.x - 1), aspect_ratio * event.pos.y / (f32)(preview_size.y - 1) }
    };
}

motion_event input_manager::get_motion_event(const rect& bounds) {
    if(event.type != motion_type::NO_MOTION &&
        bounds.tl.x <= event.pos.x && event.pos.x <= bounds.br.x &&
        bounds.tl.y <= event.pos.y && event.pos.y <= bounds.br.y) {
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

thread_pool::thread_pool() : keep_running(true) {
    u32 num_threads = std::thread::hardware_concurrency();
    threads.resize(num_threads);

    for (u32 i = 0; i < num_threads; i++) {
        threads[i] = std::thread(&thread_pool::thread_pool_loop, this, i);
    }
}

void thread_pool::thread_pool_loop(s32 i) {
    while(keep_running) {
        thread_pool_task task;

        {
            std::unique_lock<std::mutex> lock(mutex);

            if(work_queue.empty()) {
                mutex_condition.wait(lock, [&]{ return !work_queue.empty(); });
            }

            if(!keep_running) return;

            task = work_queue.front();
            work_queue.pop();
        }

        task.function(task.data);
    }
}

void thread_pool::work_on_gui_queue() {
    while(!gui_work_queue.empty()) {
        const thread_pool_task& task = gui_work_queue.front();
        gui_work_queue.pop();

        task.function(task.data);
    }
}

void thread_pool::push(thread_pool_task task) {
    {
        std::unique_lock<std::mutex> lock(mutex);
        work_queue.push(task);
    }

    mutex_condition.notify_one();
}

void thread_pool::push_gui(thread_pool_task task) {
    gui_work_queue.push(task);
}