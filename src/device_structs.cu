#include "device_structs.hpp"

__device__ void DeviceCamera::generateRay(int px, int py, int w, int h, float3 &ro, float3 &rd) const
{
    // px and py are in [0, w) and [0, h)]] pixel coordinates
    float u = (px + 0.5f) / float(w); // u ∈ [0.5/w, 1 - 0.5/w] \in [0, 1]
    float v = (py + 0.5f) / float(h); // v ∈ [0.5/h, 1 - 0.5/h] \in  [0, 1]

    float aspect = float(w) / float(h);
    float tanHalfFov = tanf(vertical_fov_ * 0.5f * M_PI / 180.0f);

    // sy = [-1, 1] * tan(fov/2); sx =  [-1, 1] * aspect * tan(fov/2)
    float sx = (2.0f * u - 1.0f) * aspect * tanHalfFov;
    float sy = (1.0f - 2.0f * v) * tanHalfFov;

    float3 dir = make_float3(
        forward_.x + sx * right_.x + sy * up_.x,
        forward_.y + sx * right_.y + sy * up_.y,
        forward_.z + sx * right_.z + sy * up_.z);
    float len = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    rd = make_float3(dir.x * len, dir.y * len, dir.z * len);
    ro = position_;
}
