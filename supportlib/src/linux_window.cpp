#ifdef LINUX
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "log.hpp"
#include "pipeline.hpp"

using namespace docscanner;

constexpr svec2 size = { 1080 / 2, 2400 / 2 };

int main() {
    if(!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW.\n");
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window;
    window = glfwCreateWindow(size.x, size.y, "Document Scanner", NULL, NULL);
    if(!window) {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize OpenGL context.\n");
        return -1;
    }

    svec2 cam_size = {};
    camera* cam = pipeline::pre_init(size, cam_size);

    pipeline_args args = {
        .preview_size = size,
        .cam_size = cam_size,
        .cam = cam,
        .enable_dark_mode = false
    };

    pipeline pipe(args);

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    do {
        pipe.render();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

    return 0;
}

#endif