#pragma once
#include <cuda_runtime.h>

struct DeviceCamera
{
    float3 forward_, up_, right_;
    float3 position_;
    float vertical_fov_;

    __device__ void generateRay(int px, int py, int w, int h, float3 &ro, float3 &rd) const;
};

struct DeviceLight
{
    float3 position;
    float3 color;
    float intensity;
    int type;
};

struct DeviceVolume
{
    cudaTextureObject_t field_tex = 0;
    cudaTextureObject_t grad_tex = 0;

    int3 dim{0, 0, 0};
    float3 voxel_size{1.f, 1.f, 1.f};
    float3 origin{0.f, 0.f, 0.f};
    float2 value_range{0.f, 1.f};
    float density_scale{1.f};
};

struct DeviceTF
{
    cudaTextureObject_t tf1D = 0;
    float2 domain{0, 1};

    __device__ float4 sample(float value) const;
};

struct DeviceScene
{
    DeviceCamera d_camera;
    DeviceVolume d_volume;
    DeviceTF d_tf;
    DeviceLight *d_lights = nullptr;
    int lights_count = 0;

    float step_size = 0.1f;
    float opacityScale = 1.0f;
    float isoValue = 0.0f;
    int mode = 0; // 0 DVR, 1 MIP, 2 ISO...
    float3 clipMin{-1e9f, -1e9f, -1e9f}, clipMax{+1e9f, +1e9f, +1e9f};
};
