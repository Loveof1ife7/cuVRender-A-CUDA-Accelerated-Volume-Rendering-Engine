#pragma once

#include <cuda_runtime.h>
#include "device_structs.hpp"

class Volume
{
public:
    struct Description
    {
        int3 dim{0, 0, 0};
        float3 origin{0.f, 0.f, 0.f};
        float3 voxelSize{1.f, 1.f, 1.f};
        float2 valueRange{0.f, 1.f};
        float densityScale{1.f};
    };

    //* hostScalar: float*，大小 dim.x*dim.y*dim.z
    Volume(const Description &desc, const float *hostScalar);
    ~Volume();

    const Description &getDesc() const { return m_desc; }
    cudaTextureObject_t getFieldTex() const { return m_fieldTex; }
    cudaTextureObject_t getGradTex() const { return m_gradTex; }
    float3 getVolumeCenter() const;

    void uploadGradient(const float3 *hostGrad);

    DeviceVolume toDevice() const;

private:
    Description m_desc;
    cudaArray_t m_arrayField = nullptr;
    cudaTextureObject_t m_fieldTex = 0;

    cudaArray_t m_arrayGrad = nullptr;
    cudaTextureObject_t m_gradTex = 0;
};

//! api usage
//* cudaArray_t : store volume data
// host data apply cudaMemcpy3D to cudaArray_t
//* cudaTextureObject_t : a texture object as volume data accesser
// bind cudaArray_t to cudaTextureObject_t

//! api usage example
// 创建3D数组仓库
//* cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
//* cudaArray* m_arrayField;
//* cudaMalloc3DArray(&m_arrayField, &channelDesc, extent);

// 将CPU数据搬运到GPU仓库
//* cudaMemcpy3DParms copyParams = {0};
//* copyParams.srcPtr = make_cudaPitchedPtr(hostData, dim.x*sizeof(float), dim.x, dim.y, dim.z);
//* copyParams.dstArray = m_arrayField;
//* copyParams.extent = extent;
//* copyParams.kind = cudaMemcpyHostToDevice;
//* cudaMemcpy3D(&copyParams);

//* cudaResourceDesc resDesc = {};
//* resDesc.resType = cudaResourceTypeArray;
//* resDesc.res.array.array = m_arrayField; // 绑定仓库

//* cudaTextureDesc texDesc = {};
//* texDesc.addressMode[0] = cudaAddressModeClamp; // 边界处理方式
//* texDesc.filterMode = cudaFilterModeLinear;     // 开启插值
//* texDesc.readMode = cudaReadModeElementType;    // 数据读取方式

//* cudaTextureObject_t m_fieldTex = 0;
//* cudaCreateTextureObject(&m_fieldTex, &resDesc, &texDesc, NULL);

//* __global__ void volumeRender(cudaTextureObject_t tex, ...) {
// 自动获得插值后的体素值
//*    float voxel = tex3D<float>(tex, x, y, z);

// 如果是梯度纹理，可以这样读取法线
//*     float3 normal = make_float3(
//*         tex3D<float>(gradTex, x, y, z).x,  // grad_x
//*         tex3D<float>(gradTex, x, y, z).y,  // grad_y
//*         tex3D<float>(gradTex, x, y, z).z   // grad_z
//*     );
//* }

// 在计算机科学中，**“句柄”（Handle）**这一术语的命名和用法源于其在实际系统中的角色和行为。以下是关于“句柄”命名的精准解释，以及为什么CUDA的`cudaTextureObject_t`被称为句柄：

// ---

// ### **1. 句柄的原始隐喻**
// **“句柄”**（Handle）直译为“把手”，其命名灵感来源于现实中的物理把手：
// - **门把手**：你无需知道门的具体构造（木材、金属、铰链机制），只需通过把手即可操作门。
// - **资源句柄**：程序无需直接操作底层资源（内存地址、硬件设备），只需通过句柄间接访问。

// 这种隐喻完美匹配CUDA纹理对象的设计：
// - 开发者无需关心纹理数据在显存中的物理布局或硬件细节，只需通过`cudaTextureObject_t`这一“把手”访问数据。

// ---

// ### **2. 句柄的技术定义**
// 在系统编程中，句柄的核心特征是：
// | 特性                | 说明                                                                 | CUDA纹理对象示例                                                                 |
// |---------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------|
// | **间接性**          | 不直接暴露资源地址，而是通过中间标识符                               | `cudaTextureObject_t`是整数ID，而非指向`cudaArray`的指针                        |
// | **抽象性**          | 隐藏底层实现细节（如内存管理、硬件配置）                             | 开发者无需知道纹理数据在显存中的物理排列方式                                    |
// | **轻量性**          | 通常为整数值或小型结构，复制/传递成本低                              | `sizeof(cudaTextureObject_t) == 4`（32位整数）                                  |
// | **安全性**          | 通过驱动/运行时验证访问权限                                          | 非法句柄会被CUDA驱动拒绝，而非导致显存非法访问                                  |
// | **稳定性**          | 底层资源变化时，句柄可保持不变                                       | 更新`cudaArray`数据后，原有`cudaTextureObject_t`仍有效                          |

// ---

// ### **3. 为什么不用“指针”或“智能指针”？**
// 虽然句柄与指针都用于访问资源，但关键差异在于：
// | 对比维度          | 指针                          | 智能指针                     | 句柄                          |
// |-------------------|-------------------------------|------------------------------|-------------------------------|
// | **直接性**        | 直接暴露内存地址               | 封装指针，但仍有地址语义      | 完全隐藏物理地址               |
// | **权限控制**      | 可任意算术操作，易越界         | 有限安全保证                 | 必须通过API操作，强制安全访问  |
// | **硬件关联**      | 无                            | 无                           | 可能绑定硬件配置（如纹理单元） |
// | **多平台一致性**  | 依赖具体内存模型               | 依赖C++实现                  | 由驱动/运行时统一管理          |

// CUDA选择“句柄”而非“指针”，正是因为需要：
// - 跨CPU/GPU的异构资源管理
// - 硬件纹理单元的抽象配置
// - 避免暴露显存物理地址

// ---

// ### **4. 句柄在CUDA中的典型生命周期**
// ```mermaid
// sequenceDiagram
//     participant CPU
//     participant Driver
//     participant GPU

//     CPU->>Driver: cudaCreateTextureObject(&handle)
//     Driver->>GPU: 分配硬件纹理单元，生成配置ID
//     Driver-->>CPU: 返回handle (e.g., 0x0001)
//     CPU->>GPU: 内核调用tex3D(handle, ...)
//     GPU->>Driver: 通过handle查找配置
//     Driver->>GPU: 按配置访问数据+硬件加速
//     GPU-->>CPU: 返回采样结果
// ```

// ---

// ### **5. 历史与跨平台印证**
// “句柄”的概念在系统编程中早有先例：
// - **Windows HGDIOBJ**：GDI对象句柄（如画笔、画布）
// - **OpenGL GLuint**：纹理/缓冲对象名称（本质是句柄）
// - **文件描述符**：Unix中访问文件的句柄（如`int fd`）

// CUDA延续了这一设计传统，其纹理句柄与上述案例有相同的设计哲学：
// - **统一接口**：无论底层是NVIDIA何种GPU架构（Ampere/Pascal），句柄用法一致
// - **安全隔离**：应用层无法通过句柄破坏驱动状态

// ---

// ### **6. 开发者如何正确理解？**
// 记住以下要点即可：
// - **句柄 ≠ 数据**：它只是访问数据的“门票”
// - **复制句柄 ≈ 复制门票**：新门票仍指向同一场“演出”（数据）
// - **销毁句柄 ≠ 销毁数据**：只是撕毁门票，演出仍可继续（其他句柄仍有效）
