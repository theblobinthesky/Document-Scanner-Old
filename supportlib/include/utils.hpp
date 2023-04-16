#pragma once
#include <cstdint>
#include <string>

#define DEBUG

#define null nullptr

#define S32_MAX 2147483647

#ifdef ANDROID
#define USES_OES_TEXTURES true
#elif defined(LINUX)
#define USES_OES_TEXTURES false
#else
#error "Unsupported platform as USES_OES_TEXTURES is not defined."
#endif

using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using s32 = int32_t;
using f32 = float;
using f64 = double;

struct vec2 {
    f32 x, y;

    vec2 operator+(vec2 other) const;
    vec2 operator-(vec2 other) const;
    vec2 operator*(f32 other) const;
    vec2 orthogonal() const;
    f32 length() const;
    f32 length_squared() const;
    vec2 normalize() const;

    static vec2 lerp(const vec2& a, const vec2& b, f32 t);
    static f32 dot(const vec2& a, const vec2& b);
    static f32 angle_between(const vec2& a, const vec2& b);
};

struct vec3 {
    f32 x, y, z;

    vec3 operator*(f32 other) const;

    static vec3 lerp(const vec3& a, const vec3& b, f32 t);
};

struct vec4 {
    f32 x, y, z, w;

    vec4 operator*(f32 other) const;

    vec4();
    vec4(f32 x, f32 y, f32 z, f32 w);
    vec4(const vec3& v);
};

struct svec2 {
    s32 x;
    s32 y;

    svec2 operator+(svec2 other) const;
    svec2 operator-(svec2 other) const;
    svec2 orthogonal() const;
    f32 length() const;
    f32 length_squared() const;
    s32 area() const;
    
    static f32 dot(const svec2& a, const svec2& b);
    static f32 angle_between(const svec2& a, const svec2& b);
};

struct rect {
    vec2 tl, br;
    vec2 middle() const;
    vec2 size() const;

    static rect from_tl_and_size(const vec2& tl, const vec2& size);
    static rect from_middle_and_size(const vec2& middle, const vec2& size);
    static rect lerp(const rect& a, const rect& b, f32 t);
};

struct uvec2 {
    u32 x;
    u32 y;
};

struct mat4 {
    f32 data[16];

    static mat4 orthographic(f32 left, f32 right, f32 bottom, f32 top, f32 near, f32 far);
};

enum class split_direction {
    HORIZONTAL, VERTICAL
};

f32 clamp(f32 val, f32 min, f32 max);

f32 random_f32(f32 min, f32 max);

f32 ease_in_sine(f32 t);

f32 ease_in_out_quad(f32 t);

f32 lerp(f32 a, f32 b, f32 t);

vec2 map_to_rect(const vec2& pt, const rect* rect);

vec3 color_from_int(s32 c);

rect cut_margins(const rect& r, f32 margin);

rect cut_margins(const rect& r, const rect& margin);

rect grid_split(const rect& r, s32 i, s32 splits, split_direction dir);

rect get_between(const rect& r, f32 t, f32 b);

#ifdef DEBUG
struct scoped_timer {
    u64 start;
    std::string name;
    scoped_timer(std::string name);
    ~scoped_timer();
};

#define SCOPED_TIMER(name) scoped_timer SCOPED_TIMER_INSTANCE(name);
#endif

#define NAMESPACE_BEGIN namespace docscanner {
#define NAMESPACE_END }