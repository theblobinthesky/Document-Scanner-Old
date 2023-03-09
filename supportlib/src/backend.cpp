#include "backend.h"

void docscanner::mat4f_load_ortho(float left, float right, float bottom, float top, float near, float far, float* mat4f) {
    float r_l = right - left;
    float t_b = top - bottom;
    float f_n = far - near;
    float tx = -(right + left) / (right - left);
    float ty = -(top + bottom) / (top - bottom);
    float tz = -(far + near) / (far - near);

    mat4f[0] = 2.0f / r_l;
    mat4f[1] = 0.0f;
    mat4f[2] = 0.0f;
    mat4f[3] = 0.0f;

    mat4f[4] = 0.0f;
    mat4f[5] = 2.0f / t_b;
    mat4f[6] = 0.0f;
    mat4f[7] = 0.0f;

    mat4f[8] = 0.0f;
    mat4f[9] = 0.0f;
    mat4f[10] = -2.0f / f_n;
    mat4f[11] = 0.0f;

    mat4f[12] = tx;
    mat4f[13] = ty;
    mat4f[14] = tz;
    mat4f[15] = 1.0f;
}
