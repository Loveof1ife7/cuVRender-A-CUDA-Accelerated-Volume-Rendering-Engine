#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "camera.hpp"
#include "cuda_utils.hpp"
#include "debug_utils.hpp"
#include "io.hpp"
#include "lights.hpp"
#include "renderer_kernel.hpp"
#include "scene.hpp"

namespace
{
struct CliOptions
{
    int width = 800;
    int height = 600;
    bool widthSet = false;
    bool heightSet = false;
    float step = 0.5f;
    float opacityScale = 1.0f;
    float densityScale = 1.0f;
    float isoValue = 0.5f;
    int mode = 0;
    int frame = 0;
    int tfSamples = 512;
    std::string output = "render.png";
    std::string vtiPath;
    std::string tfPath;
    std::string transformsPath;
};

bool parseCliArgs(int argc, char **argv, CliOptions &opt)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need = [&](int n)
        { return i + n < argc; };

        if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: cuda-volume-renderer-cli [options]\n"
                << "  --width N          Output width (default: 800)\n"
                << "  --height N         Output height (default: 600)\n"
                << "  --step F           Ray-march step size (default: 0.5)\n"
                << "  --opacity F        Opacity scale (default: 1.0)\n"
                << "  --density F        Volume density scale (default: 1.0)\n"
                << "  --iso F            Isosurface threshold (default: 0.5)\n"
                << "  --mode N           0=DVR, 1=ISO, 2=MIP (default: 0)\n"
                << "  --output PATH      Output PNG path (default: render.png)\n"
                << "  --vti PATH         Load Float32 compressed VTI volume\n"
                << "  --tf PATH          Load Vol2Splat tf_config.json\n"
                << "  --transforms PATH  Load transforms_train/test.json camera\n"
                << "  --frame N          Camera frame index for transforms JSON\n"
                << "  -raw PATH -dim X Y Z [-bpp 8|16] [-le 0|1] [-norm 0|1]\n";
            return false;
        }
        if (arg == "--width" && need(1))
        {
            opt.width = std::max(1, std::stoi(argv[++i]));
            opt.widthSet = true;
        }
        else if (arg == "--height" && need(1))
        {
            opt.height = std::max(1, std::stoi(argv[++i]));
            opt.heightSet = true;
        }
        else if (arg == "--step" && need(1))
        {
            opt.step = std::max(1e-3f, std::stof(argv[++i]));
        }
        else if (arg == "--opacity" && need(1))
        {
            opt.opacityScale = std::max(0.0f, std::stof(argv[++i]));
        }
        else if (arg == "--density" && need(1))
        {
            opt.densityScale = std::max(0.0f, std::stof(argv[++i]));
        }
        else if (arg == "--iso" && need(1))
        {
            opt.isoValue = std::stof(argv[++i]);
        }
        else if (arg == "--mode" && need(1))
        {
            opt.mode = std::stoi(argv[++i]);
        }
        else if (arg == "--output" && need(1))
        {
            opt.output = argv[++i];
        }
        else if (arg == "--vti" && need(1))
        {
            opt.vtiPath = argv[++i];
        }
        else if (arg == "--tf" && need(1))
        {
            opt.tfPath = argv[++i];
        }
        else if (arg == "--transforms" && need(1))
        {
            opt.transformsPath = argv[++i];
        }
        else if (arg == "--frame" && need(1))
        {
            opt.frame = std::max(0, std::stoi(argv[++i]));
        }
        else if (arg == "--tf-samples" && need(1))
        {
            opt.tfSamples = std::max(2, std::stoi(argv[++i]));
        }
    }
    return true;
}

std::shared_ptr<Volume> selectVolume(int argc,
                                     char **argv,
                                     const CliOptions &cli,
                                     const TransformCameraFrame &cameraFrame)
{
    if (!cli.vtiPath.empty())
    {
        VtiOptions vti;
        vti.path = cli.vtiPath;
        vti.densityScale = cli.densityScale;
        if (cameraFrame.hasVolumeWorldTransform)
        {
            vti.overrideWorldTransform = true;
            vti.worldOrigin = cameraFrame.volumeWorldOrigin;
            vti.worldSpacingScale = cameraFrame.volumeWorldSpacingScale;
        }
        return loadVtiVolume(vti);
    }

    RawOptions raw_opt;
    if (!parseRawArgs(argc, argv, raw_opt))
    {
        return makeDummyVolume();
    }
    raw_opt.densityScale = cli.densityScale;

    auto volume = loadRawVolume(raw_opt);
    if (!volume)
    {
        throw std::runtime_error("Failed to load the requested RAW volume.");
    }
    return volume;
}

Camera makeDefaultCamera(const Volume &volume, int width, int height)
{
    const auto &desc = volume.getDesc();
    const float3 center = volume.getVolumeCenter();
    const float3 extent = make_float3(
        desc.voxelSize.x * std::max(desc.dim.x - 1, 1),
        desc.voxelSize.y * std::max(desc.dim.y - 1, 1),
        desc.voxelSize.z * std::max(desc.dim.z - 1, 1));
    const float max_extent = std::max(extent.x, std::max(extent.y, extent.z));
    const Eigen::Vector3f eye(
        center.x - 1.35f * max_extent,
        center.y - 0.9f * max_extent,
        center.z + 0.65f * max_extent);

    Camera cam(eye, 45.0f, Eigen::Vector2i(width, height));
    cam.lookAt(Eigen::Vector3f(center.x, center.y, center.z), Eigen::Vector3f(0, 1, 0));
    return cam;
}

void writePng(const std::string &path, int width, int height, const std::vector<uchar4> &pixels)
{
    std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);
    for (int y = 0; y < height; ++y)
    {
        const int src_y = height - 1 - y;
        for (int x = 0; x < width; ++x)
        {
            const uchar4 px = pixels[static_cast<size_t>(src_y) * width + x];
            const size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            rgba[idx + 0] = px.x;
            rgba[idx + 1] = px.y;
            rgba[idx + 2] = px.z;
            rgba[idx + 3] = px.w;
        }
    }

    if (!stbi_write_png(path.c_str(), width, height, 4, rgba.data(), width * 4))
    {
        throw std::runtime_error("Failed to write PNG output to " + path);
    }
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        CliOptions cli;
        if (!parseCliArgs(argc, argv, cli))
        {
            return 0;
        }

        CUDA_CHECK(cudaSetDevice(0));

        TransformCameraFrame cameraFrame;
        if (!cli.transformsPath.empty())
        {
            cameraFrame = loadTransformCameraFrame(cli.transformsPath, cli.frame);
            if (cameraFrame.valid)
            {
                if (!cli.widthSet && cameraFrame.width > 0)
                    cli.width = cameraFrame.width;
                if (!cli.heightSet && cameraFrame.height > 0)
                    cli.height = cameraFrame.height;
            }
        }

        auto volume = selectVolume(argc, argv, cli, cameraFrame);
        auto tf = cli.tfPath.empty()
                      ? makeSimpleTF()
                      : loadTransferFunctionConfig(cli.tfPath, cli.tfSamples);
        Camera cam = makeDefaultCamera(*volume, cli.width, cli.height);
        if (cameraFrame.valid)
        {
            cam.SetFilmResolution(Eigen::Vector2i(cli.width, cli.height));
            cam.SetVerticalFov(cameraFrame.verticalFovDegrees);
            cam.setFrame(cameraFrame.position, cameraFrame.forward, cameraFrame.up);
        }

        Lights lights;
        std::vector<DeviceLight> host_lights(1);
        host_lights[0].position = make_float3(2, 2, 2);
        host_lights[0].color = make_float3(1, 1, 1);
        host_lights[0].intensity = 1.0f;
        host_lights[0].type = 0;
        lights.set(host_lights);

        Scene scene;
        scene.setCamera(&cam)
            .setLights(&lights)
            .setVolume(volume)
            .setTransferFunction(tf)
            .setRenderParams(cli.step, cli.opacityScale, cli.mode, cli.isoValue)
            .setClipBox(make_float3(-1e6f, -1e6f, -1e6f), make_float3(1e6f, 1e6f, 1e6f));
        scene.commit(0);

        uchar4 *d_output = nullptr;
        const size_t pixel_count = static_cast<size_t>(cli.width) * cli.height;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), pixel_count * sizeof(uchar4)));

        const dim3 block(16, 16);
        const dim3 grid(
            (cli.width + block.x - 1) / block.x,
            (cli.height + block.y - 1) / block.y);
        volumeRendererKernel<<<grid, block>>>(scene.snapshotHost(), d_output, cli.width, cli.height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<uchar4> host_output(pixel_count);
        CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, pixel_count * sizeof(uchar4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_output));

        writePng(cli.output, cli.width, cli.height, host_output);

        const SceneDebugInfo info = scene.debug();
        std::cout << "Rendered " << cli.output
                  << " at " << cli.width << "x" << cli.height
                  << " mode=" << cli.mode
                  << " step=" << cli.step
                  << " volume=(" << info.dim.x << "," << info.dim.y << "," << info.dim.z << ")\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Headless render failed: " << e.what() << "\n";
        return 1;
    }
}
