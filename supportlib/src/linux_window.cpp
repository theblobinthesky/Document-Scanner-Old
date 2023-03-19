#ifdef LINUX
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "log.hpp"
#include "pipeline.hpp"

using namespace docscanner;

constexpr uvec2 size = { 1024, 768 };

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

    pipeline pipe = {};
    svec2 cam_size = {};
    pipe.pre_init(size, &cam_size.x, &cam_size.y);
    pipe.init_backend();

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    do {
        pipe.render();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

    return 0;
}

#endif