#include "lights.hpp"
#include "cuda_utils.hpp"

void Lights::set(const std::vector<DeviceLight> &lights)
{
    const int new_count = static_cast<int>(lights.size());
    if (d_lights && new_count != m_count)
    {
        CUDA_CHECK(cudaFree(d_lights));
        d_lights = nullptr;
    }

    m_count = new_count;
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_lights), sizeof(DeviceLight) * m_count));
    }
    CUDA_CHECK(cudaMemcpy(d_lights, lights.data(), sizeof(DeviceLight) * m_count, cudaMemcpyHostToDevice));
}
