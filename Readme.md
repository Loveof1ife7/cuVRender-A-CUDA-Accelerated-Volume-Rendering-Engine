# Cuda Volume Rendering System Architecture
The architecture is designed with a clear separation between **data management** and **computational solving**, enabling modularity, scalability, and easier maintenance.

---

## 1. Core Module
Handles fundamental raymarching-based volume rendering pipeline.
The **scene** module is responsible for storing and managing all data required for rendering:

| Component           | Responsibility                                                         |
| ------------------- | ---------------------------------------------------------------------- |
| `camera`            | Constructs view rays and controls projection/perspective               |
| `transfer_function` | Handles vector/matrix operations, color mapping, and utility functions |
| `light`             | Stores and manages lighting parameters for shading models              |
| `volume`            | Represents volumetric datasets and provides sampling access            |

This separation ensures that scene configuration and dataset preparation are isolated from the rendering computation.

The **solver** module is responsible for executing the rendering process on the GPU:

| Component       | Responsibility                                                          |
| --------------- | ----------------------------------------------------------------------- |
| `volume_kernel` | CUDA kernels for volume data sampling, interpolation, and preprocessing |
| `render_kernel` | CUDA kernels for raymarching and image synthesis based on scene data    |

The solver consumes data from the **scene** and produces the final rendered image.

---

## 2. Host/Device Mirror Design

| 组件   | Host 类型（示例）          | Device 镜像         | 说明                                                 |
| ---- | -------------------- | ----------------- | -------------------------------------------------- |
| 相机   | `Camera`             | `DeviceCamera`    | 视线/基向量、位置、垂直视场；提供 `generateRay` 设备侧方法。             |
| 体数据  | `Volume`             | `DeviceVolume`    | 维度、体素尺寸、原点、值域、密度缩放；字段/梯度绑定为 `cudaTextureObject_t`。 |
| 传递函数 | `TransferFunction`   | `DeviceTF`        | 1D TF 纹理与域；设备侧 `sample(value)` 返回 `float4`（rgba）。  |
| 光源   | `std::vector<Light>` | `DeviceLight*`    | 位置/颜色/强度/类型；Scene 负责分配与拷贝到设备端数组。                   |
| 渲染参数 | `Config`/Scene 字段    | 同步到 `DeviceScene` | 步长、透明度缩放、等值面阈值、模式、裁剪盒等。                            |

### Example: Host and Device Mirror Design

**Host-side: `Volume`**
```cpp
class Volume {
public:
    struct Description {
        int3 dim;
        float3 origin;
        float3 voxelSize;
        float2 valueRange;
        float densityScale;
    };

    Volume(const Description &desc, const float *hostScalar);
    ~Volume();

    const Description &getDesc() const;
    cudaTextureObject_t getFieldTex() const;
    cudaTextureObject_t getGradTex() const;

    void uploadGradient(const float3 *hostGrad);
    DeviceVolume toDevice() const;

private:
    Description m_desc;
    cudaArray_t m_arrayField = nullptr;
    cudaTextureObject_t m_fieldTex = 0;

    cudaArray_t m_arrayGrad = nullptr;
    cudaTextureObject_t m_gradTex = 0;
};
```

**Device-side: DeviceVolume**
The device-side structs are stripped-down representations of host objects, containing only the essential data for kernel execution.
```cpp
struct DeviceVolume {
    cudaTextureObject_t field_tex = 0;
    cudaTextureObject_t grad_tex = 0;

    int3 dim{0, 0, 0};
    float3 voxel_size{1.f, 1.f, 1.f};
    float3 origin{0.f, 0.f, 0.f};
    float2 value_range{0.f, 1.f};
    float density_scale{1.f};
};
```

# Development Roadmap

| Phase   | Modules                                      | Objectives                                |
|---------|----------------------------------------------|-------------------------------------------|
| 🟢 1     | `camera`+`volume_renderer`+`film`+`utils`   | Minimal working CUDA renderer → PNG       |
| 🟡 2     | `config`+`res_manager`                       | Configurable parameters + data swapping   |
| 🟠 3     | `interpolator`+`ray`+`light`                 | Improved sampling + lighting              |
| 🔵 4     | `classifier`, `gui`, `bbox`                  | TF GUI + bounding box optimization        |
| 🟣 5     | `main_scene`, `implicit_geom`               | Scene management + implicit shapes       |

## Core principles

1. **Two-worlds, one contract**
   Treat **Host** and **Device** as two separate worlds with a **formal contract** between them. Host owns loading, lifetime, and metadata; Device sees a compact, immutable **snapshot** (struct of POD + texture/surface handles) that’s cheap to pass into kernels.

2. **Data‑oriented over OO**
   Kernels want contiguous, cache-friendly, trivially-copyable data. Prefer flat POD structs (`DeviceScene`, `DeviceVolume`, `DeviceTF`, `DeviceLight[]`) and texture objects over pointer-rich graphs.

3. **Immutable snapshots, explicit commits**
   Host objects are editable. Rendering uses a frozen **DeviceScene snapshot** created by `scene.commit()`. No hidden mutations during render. This gives determinism and easy multi-threading.

4. **Descriptors in, handles out**
   Construction uses **descriptors** (dims, spacing, origin, units, ranges). Upload returns **handles** (texture objects, device pointers) managed by RAII wrappers. Avoid exposing raw arrays.

5. **Clear frames & units**
   Be explicit about spaces: **Index (i,j,k)**, **Texture (u,v,w in \[0,1])**, **World (x,y,z)**. Store `origin`, `voxelSize`, `dim`, and value units/range. Conversions must be single-line and consistent.

6. **Consistency beats cleverness**
   One pixel format (`uchar4`), one GL internal format (`GL_RGBA8`), normalized texture coords on by default, linear filtering, border/clamp addressing. Fewer knobs → fewer bugs.

7. **Zero‑copy where it matters**
   Prefer writing directly to GL texture via surface object or PBO when mature. Until then, one device-to-device blit is fine. Don’t micro-opt prematurely—design for the *option*.

8. **Change tracking (dirty bits)**
   Every host-side component sets a dirty flag on mutation. `commit()` rebuilds only what changed: TF table? only TF; camera moved? only camera; new volume? rebuild volume block.

9. **Extensibility without recompiling kernels**
   Leave versioned fields in `DeviceScene` (e.g., `mode`, `opacityScale`, optional `gradTex`), and a `caps`/`flags` bitfield so kernels can branch safely when features exist.

10. **Graceful fallbacks**
    If optional resources are missing (no gradient texture), kernels switch to finite differences. If no lights, do emission-only. No crashing because a feature wasn’t set.

11. **Async everywhere**
    Use CUDA streams for uploads and double-buffering for time steps. `commit(stream)` so big 3D copies don’t stall the render stream.

12. **Observability**
    Build in lightweight stats: GPU mem footprint, array dims, min/max value, step counts, timings. Expose a `SceneDebugInfo`—you’ll need it.

13. **Testability**
    Make CPU reference samplers for 1D/3D textures and TF mapping. Golden tests for ray-march accumulation and TF application with fixed seeds.

# High-level architecture

* **Host layer**

  * `Volume` (RAII): owns CUDA 3D array + `cudaTextureObject_t`, metadata (dim, spacing, origin, valueRange).
  * `TransferFunction` (RAII): 1D `float4` table → 1D texture.
  * `Lights` (RAII): device buffer of `DeviceLight` (small).
  * `Camera` (host math only): exposes position + orthonormal basis + fov.
  * **Scene** (or `SceneBuilder`): references components, tracks dirty state, validates, and produces a **DeviceScene snapshot**.
  * **ResourceManager** (optional): deduplicates identical TFs/volumes, manages lifetimes across scenes.

* **Device layer**

  * `DeviceScene` (POD): `DeviceCamera`, `DeviceVolume`, `DeviceTF`, pointer to `DeviceLight[]`, counts, and render params. No STL, no Eigen, no virtual, no host-only types.

# Scene lifecycle

1. **Build**

   * Set or replace components: `scene.setVolume(vol)`, `scene.setTransferFunction(tf)`, `scene.setLights(lights)`, `scene.setCamera(&cam)`.
   * Set params: `scene.setRenderParams(step, opacityScale, mode, iso)`; `scene.setClipBox(...)`.

2. **Validate**
   On `commit()`, check invariants:

   * Volume exists if `mode` requires it.
   * TF table count > 1, domain valid, finite.
   * `stepSize` > 0 and relative to min(voxelSize).
   * Basis vectors orthonormal within tolerance.

3. **Pack**

   * Convert host camera → `DeviceCamera` (float3 only).
   * Write `DeviceVolume` from `Volume::Desc` + texture handle(s).
   * Write `DeviceTF` from TF handle + domain.
   * Attach `DeviceLight*` and count.
   * Copy to `DeviceScene` **by value** or upload to a device buffer for pointer passing.

4. **Use**

   * Pass `DeviceScene` **by value** to kernels for simplicity and ABI stability.
   * Alternatively pass a const pointer to a device-side `DeviceScene` for multi-kernel passes.

# API sketch (clean and future-proof)

```cpp
// Host-side
class Scene {
public:
    // Configuration
    Scene& setCamera(const Camera* cam);
    Scene& setVolume(std::shared_ptr<Volume> vol);
    Scene& setTransferFunction(std::shared_ptr<TransferFunction> tf);
    Scene& setLights(const Lights* lights);
    Scene& setRenderParams(float step, float opacityScale, int mode, float iso = 0.f);
    Scene& setClipBox(float3 mn, float3 mx);

    // Commit: creates an immutable snapshot for kernels
    // Optionally accepts a CUDA stream for async uploads
    void commit(cudaStream_t stream = 0);

    // Accessors for rendering
    const DeviceScene&   snapshotHost()  const; // by-value pass
    const DeviceScene*   snapshotDevice() const; // pointer pass (optional)

    // Diagnostics
    SceneDebugInfo debug() const;
};
```

Key choices here:

* **Builder-style setters** make composition readable.
* **`commit()`** is explicit and boundaries are clear. You can render older snapshots while building a new one (double buffering).
* **`snapshotHost()`** enables passing the whole struct by value to a kernel—fast and simple.
* **`snapshotDevice()`** lets you store snapshots on GPU for multi-pass pipelines.

# Memory & performance tactics

* **Textures first**: 3D scalar → linear-filtered, normalized coords. TF → linear-filtered 1D `float4`. Gradient is optional 3D `float4`.
* **Pitch and alignment**: unify pixel buffers to `uchar4`. Keep `DeviceScene` size < 4–8 KB to stay register/cache friendly when passed by value.
* **Constant memory**: tiny, truly constant things (e.g., mode enums, small LUT sizes) can live in `__constant__`; but a single by-value struct is often good enough and simpler.
* **Streams**: `commit(streamUpload)`; `render(streamRender)`. For time-varying data, ping-pong volumes/arrays per timestep.

# Extensibility hooks

* **Multi-field volumes**: make `DeviceScene` hold an array of `DeviceVolume` or a small fixed N (e.g., 2–4) and a mapping policy (`combineMode`: emission/absorption fields, bivariate TF, etc.).
* **AMR / bricked volumes**: add a bricking layer above `Volume` that assembles virtual texture coordinates; `DeviceScene` holds current brick atlas handles.
* **Masks & cut planes**: extra 3D masks as optional textures; store plane equations for slice/render modes.
* **Time series**: keep `std::vector<std::shared_ptr<Volume>>` with a `currentTimestep` and prefetch the next into a second snapshot.

# Error handling & UX

* Throw clear exceptions on resource failures with context (dims, bytes, device ID).
* Make all public methods **no-throw** except `commit()`, where allocation/copy happens.
* Provide `SceneDebugInfo` with: mem bytes, dims, value range, step size, mode, TF samples, number of lights, and a “valid” bitset.

# What this buys you

* **Determinism** (render uses frozen snapshots).
* **Simplicity** (flat structs & handles).
* **Speed** (texture sampling, by-value scene).
* **Flexibility** (optional features via flags/handles).
* **Safety** (validation + dirty-bit rebuilds).
* **Scalability** (streams, double-buffering, multi-field ready).

If you want, I can turn this into a minimal but production-ready `Scene`/`commit()` implementation skeleton with the exact method names you prefer—and a tiny kernel showing how the snapshot is consumed.

