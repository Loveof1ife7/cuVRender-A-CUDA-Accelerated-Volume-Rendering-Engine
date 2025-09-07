#include "lights.hpp"
#include "cuda_utils.hpp"

void Lights::set(const std::vector<DeviceLight> &lights)
{
    m_count = lights.size();
    if (m_count == 0)
    {
        if (d_lights)
        {
            CUDA_CHECK(cudaFree(d_lights));
            d_lights = nullptr;
        }
        return;
    }
    if (!d_lights)
    {
        CUDA_CHECK(cudaMalloc(&d_lights, sizeof(DeviceLight) * m_count));
        CUDA_CHECK(cudaMemcpy(d_lights, lights.data(), sizeof(DeviceLight) * m_count, cudaMemcpyHostToDevice));
    }
}
