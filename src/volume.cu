#include "volume.hpp"
#include "cuda_utils.hpp"
#include <vector>

//! create a 3D texture bind to a 3D volume
static cudaTextureObject_t make3DTex(cudaArray_t arr, bool normalized = true)
{
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc{};
    texDesc.normalizedCoords = normalized;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    return tex;
}

Volume::Volume(const Description &desc, const float *hostScalar) : m_desc(desc)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent ext = make_cudaExtent(m_desc.dim.x, m_desc.dim.y, m_desc.dim.z);
    CUDA_CHECK(cudaMalloc3DArray(&m_arrayField, &channelDesc, ext, cudaArrayDefault));

    cudaMemcpy3DParms cp{};
    cp.srcPtr = make_cudaPitchedPtr((void *)hostScalar,
                                    size_t(m_desc.dim.x) * sizeof(float),
                                    size_t(m_desc.dim.x), size_t(m_desc.dim.y));
    cp.dstArray = m_arrayField;
    cp.extent = ext;
    cp.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&cp));

    m_fieldTex = make3DTex(m_arrayField, true);
}
void Volume::uploadGradient(const float3 *hostGrad)
{
    cudaChannelFormatDesc channeldesc = cudaCreateChannelDesc<float4>();
    cudaExtent ext = make_cudaExtent(m_desc.dim.x, m_desc.dim.y, m_desc.dim.z);
    CUDA_CHECK(cudaMalloc3DArray(&m_arrayGrad, &channeldesc, ext, cudaArrayDefault));

    // number of channel must be 1,2,or 4
    std::vector<float4> tmp(ext.width * ext.height * ext.depth);
    for (int i = 0; i < ext.width * ext.height * ext.depth; ++i)
    {
        tmp[i] = make_float4(hostGrad[i].x, hostGrad[i].y, hostGrad[i].z, 0.f);
    }

    cudaMemcpy3DParms cp = {};
    cp.srcPtr = make_cudaPitchedPtr((void *)tmp.data(),
                                    size_t(m_desc.dim.x) * sizeof(float4),
                                    size_t(m_desc.dim.x), size_t(m_desc.dim.y));
    cp.dstArray = m_arrayGrad;
    cp.extent = ext;
    cp.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&cp));

    m_gradTex = make3DTex(m_arrayGrad, true);
}

float3 Volume::getVolumeCenter() const
{
    return make_float3(
        m_desc.origin.x + m_desc.voxelSize.x * (m_desc.dim.x - 1) * 0.5f,
        m_desc.origin.y + m_desc.voxelSize.y * (m_desc.dim.y - 1) * 0.5f,
        m_desc.origin.z + m_desc.voxelSize.z * (m_desc.dim.z - 1) * 0.5f);
}

DeviceVolume Volume::toDevice() const
{
    DeviceVolume dv{};
    dv.field_tex = m_fieldTex;
    dv.grad_tex = m_gradTex; // 可为 0
    dv.dim = m_desc.dim;
    dv.voxel_size = m_desc.voxelSize;
    dv.origin = m_desc.origin;
    dv.value_range = m_desc.valueRange;
    dv.density_scale = m_desc.densityScale;
    return dv;
}

Volume::~Volume()
{
    if (m_fieldTex)
        CUDA_CHECK(cudaDestroyTextureObject(m_fieldTex));
    if (m_arrayField)
        CUDA_CHECK(cudaFreeArray(m_arrayField));
    if (m_gradTex)
        CUDA_CHECK(cudaDestroyTextureObject(m_gradTex));
    if (m_arrayGrad)
        CUDA_CHECK(cudaFreeArray(m_arrayGrad));
}
