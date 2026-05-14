#include "debug_utils.hpp"

std::shared_ptr<Volume> makeDummyVolume()
{
    Volume::Description desc;
    desc.dim = make_int3(128, 128, 128);
    desc.origin = make_float3(0, 0, 0);
    desc.voxelSize = make_float3(1, 1, 1);
    desc.valueRange = make_float2(0, 1);
    desc.densityScale = 1.0f;

    std::vector<float> scalars(size_t(desc.dim.x) * desc.dim.y * desc.dim.z);
    // 生成一个球形密度分布
    for (int z = 0; z < desc.dim.z; ++z)
    {
        for (int y = 0; y < desc.dim.y; ++y)
        {
            for (int x = 0; x < desc.dim.x; ++x)
            {
                float nx = (x - desc.dim.x * 0.5f) / (desc.dim.x * 0.5f);
                float ny = (y - desc.dim.y * 0.5f) / (desc.dim.y * 0.5f);
                float nz = (z - desc.dim.z * 0.5f) / (desc.dim.z * 0.5f);
                float r = sqrtf(nx * nx + ny * ny + nz * nz);
                scalars[(z * desc.dim.y + y) * desc.dim.x + x] = std::max(0.f, 1.f - r);
            }
        }
    }
    return std::make_shared<Volume>(desc, scalars.data());
}

std::shared_ptr<TransferFunction> makeSimpleTF()
{
    // RGBA 梯度：黑->红->黄->白
    std::vector<float4> table;
    const int N = 256;
    table.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        float t = i / (N - 1.f);
        float3 c = make_float3(t, t * 0.5f + 0.5f * t, t); // 随意的配色
        float a = t;                                       // 越密越不透明
        table.push_back(make_float4(c.x, c.y, c.z, a));
    }
    auto tf = std::make_shared<TransferFunction>(table.data(), (int)table.size(), make_float2(0, 1));
    return tf;
}

__global__ void volumeRenderCheckCUDAGL(DeviceScene ds, uchar4 *out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float u = (x + 0.5f) / w, v = (y + 0.5f) / h;
    out[y * w + x] = make_uchar4((unsigned)(u * 255), (unsigned)(v * 255), 128, 255);
}

__global__ void kernelSampleCenter(DeviceScene ds, uchar4 *out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float s = tex3D<float>(ds.d_volume.field_tex, 0.5f, 0.5f, 0.5f);
    unsigned char g = (unsigned char)fminf(255.f, fmaxf(0.f, s * 255.f));
    out[y * w + x] = make_uchar4(0.2 * g, 0.4 * g, 0.6 * g, 255);
}

__global__ void kernelSliceUV(DeviceScene ds, uchar4 *out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float u = (x + 0.5f) / w, v = (y + 0.5f) / h;
    float s = tex3D<float>(ds.d_volume.field_tex, u, v, 0.5f);
    unsigned char g = (unsigned char)fminf(255.f, fmaxf(0.f, s * 255.f));
    out[y * w + x] = make_uchar4(0.2 * g, 0.4 * g, 0.6 * g, 255);
}

__global__ void kernelRayHit(DeviceScene ds, uchar4 *out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    float3 ro{}, rd{};
    ds.d_camera.generateRay(x, y, w, h, ro, rd);
    float3 bmin{}, bmax{};

    volumeAABB(ds.d_volume, bmin, bmax);
    float t0, t1;
    bool hit = rayBox(ro, rd, bmin, bmax, t0, t1);
    out[y * w + x] = hit ? make_uchar4(0, 255, 0, 255) : make_uchar4(255, 0, 0, 255);
}

__device__ void volumeAABB(const DeviceVolume &vol, float3 &bmin, float3 &bmax)
{
    bmin = vol.origin;
    float3 size = make_float3(
        vol.voxel_size.x * (vol.dim.x - 1),
        vol.voxel_size.y * (vol.dim.y - 1),
        vol.voxel_size.z * (vol.dim.z - 1));

    bmax = make_float3(bmin.x + size.x, bmin.y + size.y, bmin.z + size.z);
}

__device__ bool rayBox(const float3 ro, const float3 rd,
                       const float3 bmin, const float3 bmax,
                       float &tnear, float &tfar)
{
    float3 inv = make_float3(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);

    float3 t0 = make_float3(
        (bmin.x - ro.x) * inv.x,
        (bmin.y - ro.y) * inv.y,
        (bmin.z - ro.z) * inv.z);
    float3 t1 = make_float3(
        (bmax.x - ro.x) * inv.x,
        (bmax.y - ro.y) * inv.y,
        (bmax.z - ro.z) * inv.z);

    float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    tnear = fmaxf(0.0f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)));
    tfar = fminf(fminf(tmax.x, tmax.y), tmax.z);

    return tnear < tfar;
}
