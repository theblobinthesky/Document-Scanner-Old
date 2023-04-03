#include "utils.hpp"
#include <algorithm>
#include <math.h>
#include <random>

vec2 vec2::operator+(vec2 other) const {
    return { x + other.x, y + other.y };
}

vec2 vec2::operator-(vec2 other) const {
    return { x - other.x, y - other.y };
}

vec2 vec2::operator*(f32 other) const {
    return { other * x, other * y };
}

vec2 vec2::orthogonal() const {
    return { -y, x };
}

f32 vec2::length() const {
    return sqrt(length_squared());
}

f32 vec2::length_squared() const {
    return (f32)(x * x + y * y);
}

vec2 vec2::normalize() const {
    f32 len = length();
    return { x / len, y / len };
}

// todo: investigate all uses of lerp. i changed the order of a and b
vec2 vec2::lerp(const vec2& a, const vec2& b, f32 t) {
    f32 it = 1.0f - t;
    return { a.x * it + b.x * t, a.y * it + b.y * t };
}

f32 vec2::dot(const vec2& a, const vec2& b) {
    return (f32)(a.x * b.x + a.y * b.y);
}

f32 vec2::angle_between(const vec2& a, const vec2& b) {
    f32 cos_angle = std::abs(vec2::dot(a, b)) / (a.length_squared() * b.length_squared());
    return acosf(cos_angle);
}

vec3 vec3::lerp(const vec3& a, const vec3& b, f32 t) {
    return { ::lerp(a.x, b.x, t), ::lerp(a.y, b.y, t), ::lerp(a.z, b.z, t) };
}

svec2 svec2::operator+(svec2 other) const {
    return { x + other.x, y + other.y }; 
}

svec2 svec2::operator-(svec2 other) const {
    return { x - other.x, y - other.y }; 
}

svec2 svec2::orthogonal() const {
    return { -y, x };
}

f32 svec2::length() const {
    return sqrt(length_squared());
}

f32 svec2::length_squared() const {
    return (f32)(x * x + y * y);
}

s32 svec2::area() const {
    return x * y;
}

f32 svec2::dot(const svec2& a, const svec2& b) {
    return (f32)(a.x * b.x + a.y * b.y);
}

f32 svec2::angle_between(const svec2& a, const svec2& b) {
    f32 cos_angle = std::abs(svec2::dot(a, b)) / (a.length_squared() * b.length_squared());
    return acosf(cos_angle);
}

vec2 rect::middle() const {
    return (tl + br) * 0.5f;
}

vec2 rect::size() const {
    return br - tl;
}

mat4 mat4::orthographic(f32 left, f32 right, f32 bottom, f32 top, f32 near, f32 far) {
    mat4 mat = {};

    f32 r_l = right - left;
    f32 t_b = top - bottom;
    f32 f_n = far - near;
    f32 tx = -(right + left) / (right - left);
    f32 ty = -(top + bottom) / (top - bottom);
    f32 tz = -(far + near) / (far - near);

    mat.data[0] = 2.0f / r_l;
    mat.data[1] = 0.0f;
    mat.data[2] = 0.0f;
    mat.data[3] = 0.0f;

    mat.data[4] = 0.0f;
    mat.data[5] = 2.0f / t_b;
    mat.data[6] = 0.0f;
    mat.data[7] = 0.0f;

    mat.data[8] = 0.0f;
    mat.data[9] = 0.0f;
    mat.data[10] = -2.0f / f_n;
    mat.data[11] = 0.0f;

    mat.data[12] = tx;
    mat.data[13] = ty;
    mat.data[14] = tz;
    mat.data[15] = 1.0f;

    return mat;
}

f32 clamp(f32 val, f32 min, f32 max) {
    if(val < min) return min;
    if(val > max) return max;
    return val;
}

f32 random_f32(f32 min, f32 max) {
    s32 v = rand();
    f64 fv = (f64)v / (f64)(RAND_MAX - 1);
    f32 f32v = (f32)fv;

    return min + f32v * (max - min);
}

f32 ease_in_sine(f32 t) {
    return 1 - cos((t * M_PI) / 2.0f);
}

f32 ease_in_out_quad(f32 t) {
    if(t < 0.5f) {
        return 2.0f * t * t;
    } else {
        f32 x = -2.0f * t + 2.0f;
        return 1.0f - x * x / 2.0f;
    }
}

f32 lerp(f32 a, f32 b, f32 t) {
    return b * t + a * (1.0f - t);
}

vec2 map_to_rect(const vec2& pt, const rect* rect) {
    return { lerp(rect->tl.x, rect->br.x, pt.x), lerp(rect->tl.y, rect->br.y, pt.y) };
}

vec3 color_from_int(s32 c) {
    u8 r = (c >> 16) & 0xff;
    u8 g = (c >> 8) & 0xff;
    u8 b = (c >> 0) & 0xff;

    return { (f32)r / 255.0f, (f32)g / 255.0f, (f32)b / 255.0f };
}