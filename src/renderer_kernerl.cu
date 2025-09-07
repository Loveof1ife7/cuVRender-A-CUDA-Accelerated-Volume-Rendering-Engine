#include <stdio.h>
#include "cuda_utils.hpp"
#include "renderer_kernel.hpp"

using MathUtils::Kernel::add3;
using MathUtils::Kernel::clampf;
using MathUtils::Kernel::cross3;
using MathUtils::Kernel::dot3;
using MathUtils::Kernel::f3;
using MathUtils::Kernel::f4;
using MathUtils::Kernel::length3;
using MathUtils::Kernel::mad3;
using MathUtils::Kernel::mulS;
using MathUtils::Kernel::mulV;
using MathUtils::Kernel::normalize3;
using MathUtils::Kernel::sub3;

__device__ __forceinline__ float linearToSRGB(float x)
{
    x = fminf(fmaxf(x, 0.f), 1.f);
    return (x <= 0.0031308f) ? (12.92f * x) : (1.055f * powf(x, 1.f / 2.4f) - 0.055f);
}
__device__ __forceinline__ unsigned char toByte(float x)
{
    x = fminf(fmaxf(x, 0.f), 1.f);
    return (unsigned char)(x * 255.f + 0.5f);
}

__device__ bool intersectAABB(
    const float3 &ro, const float3 &rd,
    const float3 &bmin, const float3 &bmax,
    float &tmin, float &tmax)
{
    float3 inv;
    inv.x = (fabsf(rd.x) > 1e-8f) ? 1.0f / rd.x : 1e32f;
    inv.y = (fabsf(rd.y) > 1e-8f) ? 1.0f / rd.y : 1e32f;
    inv.z = (fabsf(rd.z) > 1e-8f) ? 1.0f / rd.z : 1e32f;

    const float3 t0 = mulV(sub3(bmin, ro), inv);
    const float3 t1 = mulV(sub3(bmax, ro), inv);

    // 每轴取小/大的那个（进入/离开）
    const float3 tsm = f3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    const float3 tbg = f3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

    // 三轴区间求交
    tmin = fmaxf(fmaxf(tsm.x, tsm.y), fmaxf(tsm.z, 0.0f)); // 从相机前方（t>=0）开始
    tmax = fminf(fminf(tbg.x, tbg.y), tbg.z);

    return tmax > tmin;
}
// ====== world to uvw in [0,1]======
__device__ float3 worldToUVW(const DeviceVolume &vol, const float3 &pWorld)
{
    //* vol.origin is the min corner of the volume, voxel_size is the size of a voxel, dim is the number of voxels in each direction

#ifdef DEBUG
    if (vol.dim.x <= 0 || vol.dim.y <= 0 || vol.dim.z <= 0)
    {
        printf("Volume dimension is invalid:(%d, %d, %d)\n",
               vol.dim.x, vol.dim.y, vol.dim.z);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float eps = 1e-6f;
    if (fabs(vol.voxel_size.x) < eps ||
        fabs(vol.voxel_size.y) < eps ||
        fabs(vol.voxel_size.z) < eps)
    {
        printf("Invalid voxel size: (%.6f, %.6f, %.6f)\n",
               vol.voxel_size.x, vol.voxel_size.y, vol.voxel_size.z);
        return make_float3(0.0f, 0.0f, 0.0f);
    }
#endif

    float3 inv_voxel = make_float3(
        1.0f / (vol.voxel_size.x + 1e-6f * signbit(vol.voxel_size.x)),
        1.0f / (vol.voxel_size.y + 1e-6f * signbit(vol.voxel_size.y)),
        1.0f / (vol.voxel_size.z + 1e-6f * signbit(vol.voxel_size.z)));

    float3 idx = make_float3(
        (pWorld.x - vol.origin.x) * inv_voxel.x,
        (pWorld.y - vol.origin.y) * inv_voxel.y,
        (pWorld.z - vol.origin.z) * inv_voxel.z);

    // Normalized to [0,1]
    float3 dimm1 = make_float3(
        __int2float_rn(vol.dim.x - 1),
        __int2float_rn(vol.dim.y - 1),
        __int2float_rn(vol.dim.z - 1));

    float3 uvw = make_float3(
        fminf(fmaxf(idx.x / dimm1.x, 0.0f), 1.0f),
        fminf(fmaxf(idx.y / dimm1.y, 0.0f), 1.0f),
        fminf(fmaxf(idx.z / dimm1.z, 0.0f), 1.0f));

#ifdef DEBUG
    if (isnan(uvw.x) || isnan(uvw.y) || isnan(uvw.z))
    {
        printf("Invalid UVW: pWorld(%.3f,%.3f,%.3f) -> idx(%.3f,%.3f,%.3f)\n",
               pWorld.x, pWorld.y, pWorld.z, idx.x, idx.y, idx.z);
    }
#endif
    return uvw;
}

__device__ __forceinline__ float sampleField(const DeviceVolume &vol, const float3 uvw)
{
    // volme data is scalar
    // if normalizedCoordinates=0，modify to "tex3D<float>(vol.field_tex, x, y, z)"
    return tex3D<float>(vol.field_tex, uvw.x, uvw.y, uvw.z);
}

__device__ __forceinline__ float3 sampleGradient(const DeviceVolume &vol, const float3 uvw)
{
    float4 f4 = tex3D<float4>(vol.grad_tex, uvw.x, uvw.y, uvw.z);
    return f3(f4.x, f4.y, f4.z);
}

__device__ __forceinline__ float4 sampleTF(const DeviceTF &tf, float value)
{
    // 把 value_range/domain 映射到 [0,1] 采样 1D TF
    float t = (value - tf.domain.x) / (tf.domain.y - tf.domain.x + 1e-8f);
    t = clampf(t, 0.f, 1.f);
    float4 c = tex1D<float4>(tf.tf1D, t);
    return c;
}
__device__ float sampleVolume(const float *volume, int3 dim, float3 pos)
{
    float x = pos.x * (dim.x - 1);
    float y = pos.y * (dim.y - 1);
    float z = pos.z * (dim.z - 1);

    int x0 = floor(x), x1 = min(x0 + 1, dim.x - 1);
    int y0 = floor(y), y1 = min(y0 + 1, dim.y - 1);
    int z0 = floor(z), z1 = min(z0 + 1, dim.z - 1);

    auto at = [&](int xi, int yi, int zi)
    {
        return volume[(zi * dim.y + yi) * dim.x + xi];
    };

    float c000 = at(x0, y0, z0);
    float c100 = at(x1, y0, z0);
    float c010 = at(x0, y1, z0);
    float c110 = at(x1, y1, z0);
    float c001 = at(x0, y0, z1);
    float c101 = at(x1, y0, z1);
    float c011 = at(x0, y1, z1);
    float c111 = at(x1, y1, z1);

    // trilerp(v000, v100, v010, v110, v001, v101, v011, v111, x, y, z);
    float dx = x - x0, dy = y - y0, dz = z - z0;

    float c00 = c000 * (1 - dx) + c100 * dx;
    float c10 = c010 * (1 - dx) + c110 * dx;
    float c0 = c00 * (1 - dy) + c10 * dy;

    float c01 = c001 * (1 - dx) + c101 * dx;
    float c11 = c011 * (1 - dx) + c111 * dx;
    float c1 = c01 * (1 - dy) + c11 * dy;

    c1 = c01 * (1 - dy) + c11 * dy;
    return c0 * (1 - dz) + c1 * dz;
}
__device__ __forceinline__ void compositeFrontToBack_unpremul(
    float3 rgb, float a, float4 &accum)
{
    float oneMinusAaccum = 1.f - accum.w;
    float3 rgbPremul = make_float3(rgb.x * a, rgb.y * a, rgb.z * a);
    accum.x += oneMinusAaccum * rgbPremul.x;
    accum.y += oneMinusAaccum * rgbPremul.y;
    accum.z += oneMinusAaccum * rgbPremul.z;
    accum.w += oneMinusAaccum * a;
}

__device__ __forceinline__ void compositeFrontToBack_premul(float4 sampledPremulRGBA,
                                                            float opacityScale,
                                                            float4 &accum)
{
    float a_i = clampf(sampledPremulRGBA.w * opacityScale, 0.f, 1.0f);
    float oneMinusAaccum = 1.0f - accum.w;

    // === Lacc = Lacc + (1 - alphaacc)Li
    accum.x += oneMinusAaccum * sampledPremulRGBA.x;
    accum.y += oneMinusAaccum * sampledPremulRGBA.y;
    accum.z += oneMinusAaccum * sampledPremulRGBA.z;
    accum.w += oneMinusAaccum * a_i;
}

//! ===== render kernel ====
__global__ void volumeRendererKernel(const DeviceScene scene,
                                     uchar4 *output,
                                     int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float3 ro, rd;
    scene.d_camera.generateRay(x, y, width, height, ro, rd);
    float3 bmin{}, bmax{};
    volumeAABB(scene.d_volume, bmin, bmax);

    float tmin, tmax;
    if (!rayBox(ro, rd, bmin, bmax, tmin, tmax))
    {
        output[y * width + x] = make_uchar4(0, 0, 0, 255);
        return;
    }
    const float vx = scene.d_volume.voxel_size.x;
    const float vy = scene.d_volume.voxel_size.y;
    const float vz = scene.d_volume.voxel_size.z;
    const float voxelMin = fminf(vx, fminf(vy, vz));
    float stepWorld = fmaxf(scene.step_size, 1e-4f);
    float stepInVoxel = stepWorld / fmaxf(voxelMin, 1e-6f);

    float4 accum = f4(0, 0, 0, 0);
    tmin = fmaxf(tmin, 0.0f);
    const float terminate = 0.98f;
    float maxVal = -1e30f;
    float iso = scene.isoValue;

    // ===  ray march  ===
    for (float t = tmin; t <= tmax; t += stepWorld)
    {
        float3 pWorld = add3(ro, mulS(rd, t));
        float3 uvw = worldToUVW(scene.d_volume, pWorld);

        if (uvw.x < 0.f || uvw.x > 1.f ||
            uvw.y < 0.f || uvw.y > 1.f ||
            uvw.z < 0.f || uvw.z > 1.f)
            continue;

        float s = sampleField(scene.d_volume, uvw);

        if (scene.mode == 0)
        { // -- mode 0: volume rendering
            float4 c = sampleTF(scene.d_tf, s);
            float sigma = c.w * scene.d_volume.density_scale * scene.opacityScale;
            float a_i = 1.f - expf(-sigma * stepInVoxel);
            compositeFrontToBack_unpremul(make_float3(c.x, c.y, c.z), a_i, accum);
#ifdef DEBUG
            if (x == width / 2 && y == height / 2)
            {
                printf("a_i=%.3e step=%.3e accum=(%.3e,%.3e,%.3e,%.3e)\n",
                       a_i, stepWorld, accum.x, accum.y, accum.z, accum.w);
            }
#endif

            if (accum.w > terminate)
                break;
        }
        else if (scene.mode == 1)
        { // -- mode 1: isosurface rendering
        }
        else if (scene.mode == 2)
        { // -- mode 2: MIP
            maxVal = fmaxf(maxVal, s);

            continue;
        }
    }

    // write output
    float3 color = (accum.w > 1e-6f)
                       ? make_float3(accum.x / accum.w, accum.y / accum.w, accum.z / accum.w)
                       : make_float3(0, 0, 0);
#ifdef DEBUG
    if (x == width / 2 && y == height / 2)
    {
        unsigned char R = toByte(color.x);
        unsigned char G = toByte(color.y);
        unsigned char B = toByte(color.z);
        printf("final color linear=(%.3f,%.3f,%.3f) -> 8bit=(%u,%u,%u)\n",
               color.x, color.y, color.z, R, G, B);
    }
#endif

    if (scene.mode == 0)
    {
        // Total light intensity divided by total opacity
        color = accum.w > 1e-6f ? f3(accum.x / accum.w, accum.y / accum.w, accum.z / accum.w) : f3(0.f, 0.f, 0.f);

        // sRGB 编码后写回
        output[y * width + x] = make_uchar4(
            toByte(linearToSRGB(color.x)),
            toByte(linearToSRGB(color.y)),
            toByte(linearToSRGB(color.z)),
            255);
        return;
    }
}

__global__ void kernelFirstSample(const DeviceScene scene,
                                  uchar4 *output,
                                  int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float3 ro{}, rd{};
    scene.d_camera.generateRay(x, y, width, height, ro, rd);

    float3 bmin{}, bmax{};
    volumeAABB(scene.d_volume, bmin, bmax);

    float tmin, tmax;
    if (!rayBox(ro, rd, bmin, bmax, tmin, tmax))
    {
        output[y * width + x] = make_uchar4(0, 0, 0, 255);
        return;
    }

    float4 accum = f4(0, 0, 0, 0);
    float stepWorld = 1e-3f;
    tmin = fmaxf(tmin, 0.0f);

    // first sample
    float3 p_in = add3(ro, mulS(rd, tmin + stepWorld));
    float3 uvw_in = worldToUVW(scene.d_volume, p_in);
    float s_in = sampleField(scene.d_volume, uvw_in);

    // middle samples
    float3 p_mid = add3(
        ro,
        mulS(rd, (fmaxf(tmin, 0.f) + tmax) * 0.5f));
    float3 uvw_mid = worldToUVW(scene.d_volume, p_mid);
    float s_mid = sampleField(scene.d_volume, uvw_mid);

    unsigned char g = (unsigned char)(fminf(1.f, s_mid) * 255);
    output[y * width + x] = make_uchar4(g, g, g, 255);

    if (x == width / 2 && y == height / 2)
    {
        printf("t0=%.4f t1=%.4f t_in=%.4f uvw_in=(%.3f,%.3f,%.3f) s_in=%.4f\n",
               tmin, tmax, tmin + stepWorld, uvw_in.x, uvw_in.y, uvw_in.z, s_in);
        printf("t_mid=%.4f uvw_mid=(%.3f,%.3f,%.3f) s_mid=%.4f\n",
               tmin + (fmaxf(tmin, 0.f) + tmax) * 0.5f, uvw_mid.x, uvw_mid.y, uvw_mid.z, s_mid);
    }

    return;
}

__global__ void kernelMip(const DeviceScene scene,
                          uchar4 *output,
                          int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const int idx = y * width + x;

    float3 ro{}, rd{};
    scene.d_camera.generateRay(x, y, width, height, ro, rd);

    float tmin{}, tmax{};
    if (!intersectAABB(ro, rd, scene.clipMin, scene.clipMax, tmin, tmax))
    {
        output[idx] = make_uchar4(0, 0, 0, 255);
    }
    // step size
    float vx = scene.d_volume.voxel_size.x, vy = scene.d_volume.voxel_size.y, vz = scene.d_volume.voxel_size.z;
    float step = fmaxf(fminf(vx, fminf(vy, vz)) * scene.step_size, 1e-4f);

    float vmax = 0.f;
    for (float t = fmaxf(tmin, 0.f); t <= tmax; t += step)
    {
        float3 p = make_float3(ro.x + rd.x * t, ro.y + rd.y * t, ro.z + rd.z * t);
        float3 uvw_in = worldToUVW(scene.d_volume, p);
        float s_in = tex3D<float>(scene.d_volume.field_tex, uvw_in.x, uvw_in.y, uvw_in.z);
        vmax = fmaxf(vmax, s_in);
    }
    unsigned char g = (unsigned char)(fminf(1.f, vmax) * 255);
    output[y * width + x] = make_uchar4(g, g, g, 255);
}