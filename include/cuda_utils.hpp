#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __FILE__, __LINE__)

inline void cudaCheck(cudaError_t e, const char *what, const char *file, int line)
{
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e) +
                                 " @ " + file + ":" + std::to_string(line));
    }
}

#define CHECK_CUDA_IN_HOST(call)                                             \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                      << __FILE__ << ":" << __LINE__ << "\n";                \
            exit(1);                                                         \
        }                                                                    \
    }
/**
 * Unreachable statement
 */

#define UNREACHABLE                                                            \
    std::cout << "Error: Unreachable code executed. Exit(-1)..." << std::endl; \
    exit(-1);

/**
 * String processing utilities
 */

namespace StrUtils
{
    /* Trim from left */
    __host__ void ltrim(std::string &s);
    /* Trim from right */
    __host__ void rtrim(std::string &s);
    /* Trim from both left and right */
    __host__ void trim(std::string &s);
    /* Starts with */
    __host__ bool startsWith(const std::string &s, const std::string &prefix);
};

/**
 * Mathematical utilities
 */

/**
 * Utilities for checking CUDA errors
 */

#define SINGLE_THREAD                        \
    if (threadIdx.x != 0 || blockIdx.x != 0) \
    {                                        \
        return;                              \
    }

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

__host__ void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line);

template <typename T>
__host__ T *cudaToDevice(T *h_ptr, int num = 1)
{
    T *d_ptr;
    checkCudaErrors(cudaMallocManaged((void **)&d_ptr, sizeof(T) * num));
    checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, sizeof(T) * num, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    return d_ptr;
}

/**
 * Image processing utilities
 */

namespace ImageUtils
{
    /* Performs Gamma correction on x and outputs an integer between 0-255. */
    __host__ __device__ unsigned char gammaCorrection(float x);
    /* Converts RGB pixel array to byte array with 4 channels (RGBA) */
    __global__ void pixelArrayToBytesRGBA(Eigen::Vector3f *pix_arr, unsigned char *bytes, int res_x, int res_y);
};

namespace MathUtils::Camera
{
    static inline float3 f3(const Eigen::Vector3f &v) { return make_float3(v.x(), v.y(), v.z()); }
}

namespace MathUtils::Kernel
{
    __device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
    __device__ __forceinline__ float4 f4(float x, float y, float z, float w) { return make_float4(x, y, z, w); }
    __device__ __forceinline__ float dot3(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    __device__ __forceinline__ float3 cross3(const float3 &a, const float3 &b)
    {
        return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    __device__ __forceinline__ float length3(const float3 &v) { return sqrtf(dot3(v, v)); }
    __device__ __forceinline__ float3 normalize3(const float3 &v)
    {
        float l = length3(v);
        return l > 0.f ? f3(v.x / l, v.y / l, v.z / l) : f3(0, 0, 0);
    }
    __device__ __forceinline__ float3 add3(const float3 &a, const float3 &b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __device__ __forceinline__ float3 sub3(const float3 &a, const float3 &b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
    __device__ __forceinline__ float3 mulS(const float3 &a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
    __device__ __forceinline__ float3 mulV(const float3 &a, const float3 &b) { return f3(a.x * b.x, a.y * b.y, a.z * b.z); }
    __device__ __forceinline__ float3 mad3(const float3 &a, float s, const float3 &b)
    { // a + s*b
        return f3(a.x + s * b.x, a.y + s * b.y, a.z + s * b.z);
    }
    __device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

};

namespace MathUtils::Common
{
    /* Clamps the input x to the closed range [lo, hi] */
    __host__ __device__ float clamp(float x, float lo, float hi);
    /* Guassian function */
    __host__ __device__ float Gaussian(float mu, float sigma, float x);
};
