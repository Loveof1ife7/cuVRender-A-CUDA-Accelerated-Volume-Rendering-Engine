#include "scene.hpp"
#include "volume_renderer.hpp"
#include "renderer_kernel.hpp"
#include "debug_utils.hpp"
#include "cuda_utils.hpp"
#include "cuda_gl_interop.h"

static void glCreateRGBA8Texture(GLuint &tex, int w, int h)
{
    if (!tex)
        glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // 分配显存但不上传数据（最后我们要用 CUDA 来写这块内存）
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

VolumeRenderer::VolumeRenderer(const CreateInfo &ci)
    : width_(ci.width), height_(ci.height), useSurfaceWrite_(ci.useSurfaceWrite)
{
    // 1) create gl texture as
    createGLTexture(width_, height_);

    // 2) register gl texture to cuda
    registerCudaInterop();

    // 3) allocate device memory for kernel's output
    if (!useSurfaceWrite_)
        allocDeviceBuffer();
}

void VolumeRenderer::createGLTexture(int w, int h) { glCreateRGBA8Texture(glTex_, w, h); }

void VolumeRenderer::destoryGLTexture()
{
    if (glTex_)
    {
        glDeleteTextures(1, &glTex_);
        glTex_ = 0;
    }
}

void VolumeRenderer::registerCudaInterop()
{
    // Register flags:
    // -WriteDiscard: linear buffered path usage; We write the target texture with full coverage per frame
    // -SurfaceLoadStore: Needs this flag if you're going to surface
    unsigned int flags = useSurfaceWrite_ ? cudaGraphicsRegisterFlagsSurfaceLoadStore
                                          : cudaGraphicsRegisterFlagsWriteDiscard;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaRes_, glTex_, GL_TEXTURE_2D, flags));
}

void VolumeRenderer::unregisterCudaInterop()
{
    if (cudaRes_)
    {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cudaRes_));
        cudaRes_ = nullptr;
    }
}

void VolumeRenderer::allocDeviceBuffer()
{
    if (!d_output_)
    {
        CUDA_CHECK(
            cudaMalloc(&d_output_, size_t(width_) * size_t(height_) * sizeof(uchar4)));
    }
}

void VolumeRenderer::freeDeviceBuffer()
{
    if (d_output_)
    {
        CUDA_CHECK(cudaFree(d_output_));
        d_output_ = nullptr;
    }
}

VolumeRenderer::~VolumeRenderer()
{
    freeDeviceBuffer();
    destoryGLTexture();
    unregisterCudaInterop();
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    };
}

void VolumeRenderer::resize(int newW, int newH)
{
    if (newW == width_ && newH == height_)
        return;
    width_ = newW;
    height_ = newH;

    unregisterCudaInterop();
    destoryGLTexture();
    createGLTexture(width_, height_);
    registerCudaInterop();

    if (!useSurfaceWrite_)
    {
        freeDeviceBuffer();
        allocDeviceBuffer();
    }
}
void VolumeRenderer::ensureCapacity(int w, int h)
{
}

//! [ CUDA Kernel ]
//!     │
//!     ↓ (写入)
//! [ d_output_ ] → (可选) cudaMemcpy → [ cudaRes_ (映射到 glTex_) ] → OpenGL渲染
//!     │                                   ▲
//!     └───────────────────────────────────┘ (当 useSurfaceWrite_=true 时直接写入)
void VolumeRenderer::render(Scene &scene, bool commitScene)
{
    if (commitScene)
        scene.commit(stream_);

    //* cudaRes_ has binded to the gl texture "glTex_" in cudaGraphicsGLRegisterImage()
    //* cudaGraphicsMapResources 临时将 OpenGL 纹理的控制权转移给 CUDA，
    //* cudaRes_
    // 是一个 cudaGraphicsResource*，代表注册好的 OpenGL 资源的“全局句柄”。
    // 它不直接指向 GPU 像素内存
    // 只是 CUDA 与 GL 之间的“资源注册对象”，用来做 map/unmap 这样的生命周期管理。
    // 在注册 (cudaGraphicsGLRegisterImage) 和映射 (cudaGraphicsMapResources) 时用它。
    //* cudaArray_t dstArray
    // 是 CUDA 端的真正的图像内存对象句柄，指向这个 GL 纹理在 GPU 上的实际存储（CUDA array）。
    // 只有在 cudaGraphicsMapResources 之后，调用 cudaGraphicsSubResourceGetMappedArray 才能得到。
    // 用它才能直接访问像素数据（拷贝、创建 texture/surface 对象等）。
    // 如果没这个，CUDA kernel 就没办法读写 GL 纹理的存储。
    cudaGraphicsMapResources(1, &cudaRes_, stream_);
    cudaArray_t dstArray = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&dstArray, cudaRes_, 0, 0);

    // prepare kernal params, pass by value
    const DeviceScene &ds = scene.snapshotHost();

    dim3 blockSize(16, 16);
    dim3 gridSize((width_ + blockSize.x - 1) / blockSize.x,
                  (height_ + blockSize.y - 1) / blockSize.y);

    if (!useSurfaceWrite_)
    {
        ensureCapacity(width_, height_);

        volumeRendererKernel<<<gridSize, blockSize, 0, stream_>>>(ds, d_output_, width_, height_);
        //  volumeRenderCheckCUDAGL<<<gridSize, blockSize, 0, stream_>>>(ds, d_output_, width_, height_);
        //  kernelSampleCenter<<<gridSize, blockSize, 0, stream_>>>(ds, d_output_, width_, height_);
        //  kernelSliceUV<<<gridSize, blockSize, 0, stream_>>>(ds, d_output_, width_, height_);
        // kernelRayHit<<<gridSize, blockSize, 0, stream_>>>(ds, d_output_, width_, height_);
        // kernelFirstSample<<<gridSize, blockSize, 0, stream_>>>(ds, d_output_, width_, height_);
        // kernelMip<<<gridSize, blockSize, 0, stream_>>>(ds, d_output_, width_, height_);
        CUDA_CHECK(cudaGetLastError());

        // 把线性缓冲拷到 GL 纹理的 cudaArray（设备到设备拷贝）
        CUDA_CHECK(cudaMemcpy2DToArrayAsync(
            dstArray,
            /*wOffset*/ 0, /*hOffset*/ 0,
            /*src*/ d_output_,
            /*srcPitch*/ size_t(width_) * sizeof(uchar4),
            /*widthInBytes*/ size_t(width_) * sizeof(uchar4),
            /*height*/ size_t(height_),
            cudaMemcpyDeviceToDevice,
            stream_));
    }
    else
    {
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
    // The CUDA occupation of an OpenGL texture is relieved, allowing OpenGL to use the texture again
    cudaGraphicsUnmapResources(1, &cudaRes_, stream_);
}

//! cuda-opengl interoperation
//!    [ OpenGL 纹理对象 ] living room
//    m_output_texture (GLuint)， only belong to OpenGL
//              │
//              │ 注册（cudaGraphicsGLRegisterImage）
//              ▼
//!  [ CUDA 图形资源句柄 ] key
//  m_cuda_output_resource (cudaGraphicsResource*) , is not a real CUDA resource, but a handle to the OpenGL texture, establishing the resource sharing channel between OpenGL and CUDA
//              │
//              │ 映射后访问（cudaGraphicsSubResourceGetMappedArray）
//              ▼
//!    [ CUDA 可写数组视图 ] space behind the opening door
//    m_cuda_array (cudaArray_t) , cuda view of the OpenGL texture, used for writing data in CUDA kernel
