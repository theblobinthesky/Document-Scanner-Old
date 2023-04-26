#pragma once
#include "utils.hpp"

#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>

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

typedef void(*thread_pool_function)(void*);

struct thread_pool_task {
    thread_pool_function function;
    void* data;
};

struct thread_pool {
    bool keep_running;
    std::vector<std::thread> threads;
    
    std::condition_variable mutex_condition;
    std::mutex mutex;

    std::queue<thread_pool_task> work_queue;
    std::queue<thread_pool_task> gui_work_queue;

    thread_pool();
    void thread_pool_loop(s32 i);
    void work_on_gui_queue();
    void push(thread_pool_task task); 
    void push_gui(thread_pool_task task);
};

struct camera;

camera* find_and_open_back_camera(const svec2& min_size, svec2& size);

void resume_camera_capture(camera* cam);
void pause_camera_capture(const camera* cam);
void get_camera_frame(const camera* cam);

struct file_context;

void read_from_package(file_context* ctx, const char* path, u8* &data, u32 &size);
void read_from_internal_file(file_context* ctx, const char* path, u8* &data, u32 &size);
void write_to_internal_file(file_context* ctx, const char* path, u8* data, u32 size);

NAMESPACE_END

#ifdef ANDROID
#include <jni.h>

NAMESPACE_BEGIN

void platform_init(JNIEnv *env, jobject obj, jobject asset_mngr, jobject surface, jstring internal_data_path, jint preview_width, jint preview_height, jboolean enable_dark_mode);
void platform_destroy(JNIEnv* env, jobject obj);
void platform_motion_event(JNIEnv* env, jobject obj, jint event, jfloat x, jfloat y);
void platform_render(JNIEnv *env, jobject obj);

NAMESPACE_END
#endif
