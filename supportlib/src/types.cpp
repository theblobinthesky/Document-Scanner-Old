#include "types.hpp"
#include <algorithm>
#include <math.h>

svec2 svec2::operator-(svec2 other) const {
    return { x - other.x, y - other.y }; 
}

f32 svec2::length() const {
    return sqrt(length_squared());
}

f32 svec2::length_squared() const {
    return (f32)(x * x + y * y);
}

f32 svec2::dot(const svec2& a, const svec2& b) {
    return (f32)(a.x * b.x + a.y * b.y);
}

f32 svec2::angle_between(const svec2& a, const svec2& b) {
    f32 cos_angle = std::abs(svec2::dot(a, b)) / (a.length_squared() * b.length_squared());
    return acosf(cos_angle);
}