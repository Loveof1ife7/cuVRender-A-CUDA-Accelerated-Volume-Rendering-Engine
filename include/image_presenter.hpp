#pragma once

#include <glad/glad.h>

class ImagePresenter
{
public:
    ImagePresenter(int width, int height);
    ~ImagePresenter();

    void display(GLuint textureID);

private:
    GLuint vao,
        vbo;
    GLuint shaderProgram;
    int screenWidth, screenHeight;
    GLuint compileShader(const char *source, GLenum type);
    GLuint createShaderProgram(const char *vs_src, const char *fs_src);
};