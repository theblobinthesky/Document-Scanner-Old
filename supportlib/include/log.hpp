#pragma once
#include <csignal>

#ifdef ANDROID
#include <android/log.h>

#define LOG_TAG "docscanner"
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#elif defined(LINUX)
#include <stdio.h>

#define LOGE(...) { fprintf(stderr, "ERROR: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); }
#define LOGI(...) { printf("INFO: "); printf(__VA_ARGS__); printf("\n"); }

#endif

#define BREAK() raise(SIGTRAP)
#define LOGE_AND_BREAK(...) LOGE(__VA_ARGS__); BREAK()
#define ASSERT(working_condition, ...) if (!(working_condition)) { LOGE_AND_BREAK(__VA_ARGS__); }