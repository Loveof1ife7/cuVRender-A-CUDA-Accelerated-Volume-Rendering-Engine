// transfer_function.cpp
#include "transfer_function.hpp"
#include "cuda_utils.hpp"

static cudaTextureObject_t make1DTex(cudaArray_t arr)
{
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc td{};
    td.normalizedCoords = 1;
    td.filterMode = cudaFilterModeLinear;
    td.addressMode[0] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t t = 0;
    CUDA_CHECK(cudaCreateTextureObject(&t, &res, &td, nullptr));
    return t;
}

TransferFunction::TransferFunction(const float4 *table, int count, float2 domain)
    : m_count(count), m_domain(domain)
{
    CUDA_CHECK(cudaMalloc(&m_device_table, count * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(m_device_table, table, count * sizeof(float4), cudaMemcpyHostToDevice));

    // description of the array storing float4  rgba
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();

    // allocate the 1dim cuda array
    CUDA_CHECK(cudaMallocArray(&m_array, &ch, count, 1, cudaArrayDefault));

    // using cudaMemcpy2DToArray: height = 1
    CUDA_CHECK(cudaMemcpy2DToArray(m_array, 0, 0, table, count * sizeof(float4),
                                   count * sizeof(float4), 1, cudaMemcpyHostToDevice));

    // binding cudaArrray_t to a cudaTextureObject_t
    m_tex = make1DTex(m_array);
}
TransferFunction::~TransferFunction()
{
    if (m_tex)
        CUDA_CHECK(cudaDestroyTextureObject(m_tex));
    if (m_array)
        CUDA_CHECK(cudaFreeArray(m_array));
    if (m_device_table)
        CUDA_CHECK(cudaFree(m_device_table));
}

// texture query
__device__ float4 TransferFunction::sample(float x) const
{
    float normalized_x = (x - m_domain.x) / (m_domain.y - m_domain.x);
    return tex1D<float4>(m_tex, normalized_x);
}

// directly query
__device__ float4 TransferFunction::lookup(float x) const
{
    float normalized_x = (x - m_domain.x) / (m_domain.y - m_domain.x);
    int idx = min(max(0, (int)(normalized_x * (m_count - 1))), m_count - 1);
    return m_device_table[idx];
}

DeviceTF TransferFunction::toDevice() const
{
    DeviceTF d_tf{};
    d_tf.domain = m_domain;
    d_tf.tf1D = m_tex;

    return d_tf;
}

//! api usage
// __global__ void renderKernel(
//     float *scalar_field,   // input scalar volume
//     TransferFunction *tf,  // transfer function
//     uchar4 *output,        // output image
//     int width, int height, // output image resolution
//     float data_range       // scalar volume data range
// )
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= width || y >= height)
//         return;

//     float scalar_value = scalar_field[y * width + x];
//     float normalized = (scalar_value - data_range.x) / (data_range.y - data_range.x);

//!     float4 color = tex1D<float4>(tf->getCudaTex(), normalized);

// }