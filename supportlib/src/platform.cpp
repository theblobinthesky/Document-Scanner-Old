#include "platform.hpp"

using namespace docscanner;

// todo: something related to threading still seems to produce errors from time to time shortly after startup

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

        LOGI("issued thread task %lu on %d", (u64)task.data, i);

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
    LOGI("issued thread push");

    {
        std::unique_lock<std::mutex> lock(mutex);
        work_queue.push(task);
    }

    mutex_condition.notify_one();
}

void thread_pool::push_gui(thread_pool_task task) {
    gui_work_queue.push(task);
}