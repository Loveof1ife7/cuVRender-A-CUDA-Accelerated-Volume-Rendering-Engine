#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "device_structs.hpp"

class Lights
{
public:
    Lights() = default;
    ~Lights()
    {
        if (d_lights)
            cudaFree(d_lights);
    }
    void set(const std::vector<DeviceLight> &lights);

    __host__ __device__ DeviceLight *getDevicePointer() const { return d_lights; }
    __host__ __device__ int count() const { return m_count; }

private:
    DeviceLight *d_lights = nullptr;
    int m_count = 0;
};