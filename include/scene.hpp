// scene.hpp
#pragma once
#include <memory>
#include <optional>
#include "device_structs.hpp"
#include "camera.hpp"
#include "volume.hpp"
#include "transfer_function.hpp"
#include "lights.hpp"

struct SceneDebugInfo
{
    bool valid = false;
    int3 dim{0, 0, 0};
    float3 voxelSize{1, 1, 1};
    float2 valueRange{0, 1};
    int tfSize = 0;
    int lightCount = 0;
    float stepSize = 0.f;
    int mode = 0;
};

class Scene
{
public:
    Scene() = default;
    ~Scene();

    Scene &setCamera(const Camera *cam);
    Scene &setLights(const Lights *lights);
    Scene &setVolume(std::shared_ptr<Volume> volume);
    Scene &setTransferFunction(std::shared_ptr<TransferFunction> tf);
    Scene &setRenderParams(float step_size, float opacity_scale, int mode, float iso_value);
    Scene &setClipBox(float3 clip_min, float3 clip_max);

    //* validation
    void validateOrThrow() const;

    //* commit snapshoot
    void commit(cudaStream_t stream);

    const DeviceScene &snapshotHost() const { return m_ds_host; }
    const DeviceScene *snapshotDevice() const { return m_ds_dev; }

    SceneDebugInfo debug() const;

private:
    //*  host data manager
    const Camera *m_cam = nullptr;
    const Lights *m_lights = nullptr;

    std::shared_ptr<Volume> m_volume;
    std::shared_ptr<TransferFunction> m_tf;

    //*  rendering parameters
    float m_stepSize = 0.5f;
    float m_opacityScale = 1.0f;
    int m_mode = 0;
    float m_isoValue = 0.0f;
    float3 m_clipMin{-1e9f, -1e9f, -1e9f}, m_clipMax{+1e9f, +1e9f, +1e9f};

    //*  device data
    // snapshoot
    DeviceScene m_ds_host{};
    DeviceScene *m_ds_dev = nullptr;

    //*  dirty mark
    bool m_dirtyCam = true, m_dirtyVol = true, m_dirtyTF = true, m_dirtyLights = true, m_dirtyParams = true;
};

//! api usage
// Scene scene;
// scene.setCamera(&camera)
//      .setVolume(volumePtr)
//      .setTransferFunction(tfPtr)
//      .setLights(&lights)
//      .setRenderParams(0.5f, 1.0f, /*mode*/0)
//      .setClipBox(make_float3(-1e9f), make_float3(1e9f));
// scene.commit(); // 可给stream

// // 渲染
// auto& ds = scene.snapshotHost();
// volumeRendererKernel<<<grid,block>>>(ds, d_output, width, height);
