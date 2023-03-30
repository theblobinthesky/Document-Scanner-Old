#include "types.hpp"
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

vec2 vec2::lerp(const vec2& a, const vec2& b, f32 t) {
    f32 it = 1.0f - t;
    return { a.x * t + b.x * it, a.y * t + b.y * it };
}

f32 vec2::dot(const vec2& a, const vec2& b) {
    return (f32)(a.x * b.x + a.y * b.y);
}

f32 vec2::angle_between(const vec2& a, const vec2& b) {
    f32 cos_angle = std::abs(vec2::dot(a, b)) / (a.length_squared() * b.length_squared());
    return acosf(cos_angle);
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

vec2 svec2::lerp(const svec2& a, const svec2& b, f32 t) {
    f32 it = 1.0f - t;
    return { a.x * t + b.x * it, a.y * t + b.y * it };
}

f32 svec2::dot(const svec2& a, const svec2& b) {
    return (f32)(a.x * b.x + a.y * b.y);
}

f32 svec2::angle_between(const svec2& a, const svec2& b) {
    f32 cos_angle = std::abs(svec2::dot(a, b)) / (a.length_squared() * b.length_squared());
    return acosf(cos_angle);
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