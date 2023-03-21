#include "types.hpp"
#include <algorithm>
#include <math.h>

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