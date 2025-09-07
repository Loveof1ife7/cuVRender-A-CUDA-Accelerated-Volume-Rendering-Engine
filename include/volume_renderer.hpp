#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include <glad/glad.h>

#include "scene.hpp"
#include "camera.hpp"
#include "lights.hpp"

class VolumeRenderer
{
public:
    struct CreateInfo
    {
        int width = 1;
        int height = 1;
        bool useSurfaceWrite = false; // if true, kernel writes directly to GL texture via surface

        CreateInfo(int w, int h, bool useSurface)
            : width(w), height(h), useSurfaceWrite(useSurface)
        {
        }
    };

    explicit VolumeRenderer(const CreateInfo &ci);
    ~VolumeRenderer();

    void render(Scene &scene, bool commitScene = false);

    // Accessors
    GLuint getResultTexture() const { return glTex_; }
    int width() const { return width_; }
    int height() const { return height_; }

    void resize(int newW, int newH);

    void setStream(cudaStream_t stream) { stream_ = stream; }
    void setUseSurfaceWrite(bool on) { useSurfaceWrite_ = on; }
    bool useSurfaceWrite() const { return useSurfaceWrite_; }

private:
    void createGLTexture(int w, int h); // create a gl_rbga8
    void destoryGLTexture();

    void registerCudaInterop(); // register gl_rbga8 to cuda, get cudaGraphicsResource*）
    void unregisterCudaInterop();

    void allocDeviceBuffer(); //  allocate device buffer
    void freeDeviceBuffer();

    void ensureCapacity(int w, int h);

private:
    int width_, height_;
    bool useSurfaceWrite_ = false;

    // —— OpenGL texture object
    GLuint glTex_{0};

    // —— CUDA-GL interop handle, allow cuda to access gl texture
    cudaGraphicsResource *cudaRes_ = nullptr; // registered gl texture

    // kernal writing buffer
    uchar4 *d_output_{nullptr}; // device linear buffer

    // Stream (shared for kernel + copies)
    cudaStream_t stream_ = 0; // default stream
};