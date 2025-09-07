#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "device_structs.hpp"
#include "debug_utils.hpp"
// ====== Ray-AABB intersection ======

__device__ bool intersectAABB(
    const float3 &ro, const float3 &rd,
    const float3 &bmin, const float3 &bmax,
    float &tmin, float &tmax);

// ====== world to uvw in [0,1]======
__device__ float3 worldToUVW(const DeviceVolume &vol, const float3 &pWorld);

__device__ __forceinline__ float sampleField(const DeviceVolume &vol, const float3 uvw);

__device__ __forceinline__ float3 sampleGradient(const DeviceVolume &vol, const float3 uvw);

__device__ __forceinline__ float4 sampleTF(const DeviceTF &tf, float value);

__device__ __forceinline__ void compositeFrontToBack(float4 sampleRGBA, float opacityScale, float4 &accum);

__global__ void volumeRendererKernel(const DeviceScene scene,
                                     uchar4 *output,
                                     int width, int height);

__global__ void kernelFirstSample(DeviceScene ds, uchar4 *out, int width, int height);

__global__ void kernelMip(const DeviceScene scene, uchar4 *output, int width, int height);