#pragma once
#include <cuda_runtime.h>
#include "device_structs.hpp"

class TransferFunction
{
public:
    TransferFunction() = default;
    TransferFunction(const float4 *data, int count, float2 domain = {0, 1});
    ~TransferFunction();

    __device__ float4 sample(float x) const;

    __device__ float4 lookup(float x) const;

    __host__ __device__ cudaTextureObject_t getCudaTex() const { return m_tex; }
    __host__ __device__ float2 getDomain() const { return m_domain; }

    DeviceTF toDevice() const;

private:
    int m_count = 0;
    float2 m_domain = {0, 1};
    float4 *m_device_table = nullptr;
    cudaArray_t m_array = nullptr;
    cudaTextureObject_t m_tex = 0;
};
