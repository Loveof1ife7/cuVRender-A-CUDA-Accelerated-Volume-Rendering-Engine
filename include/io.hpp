#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <limits>
#include <memory>
#include "cuda_runtime.h"

#include "volume.hpp"

// 解析命令行参数：
// 例：
//   ./CudaSciVis -raw ~/data/engine.raw -dim 256 256 256 -bpp 16 -le 1 -norm 1 -vox 1 1 1 -org 0 0 0 -dens 1.0
//   ./CudaSciVis -raw head256.raw -dim 256 256 256

struct RawOptions
{
    std::string path;
    int3 dim{0, 0, 0};
    int bpp = 8;              // 8 or 16-bit
    bool littleEndian = true; // 16-bit
    bool normalize = true;    // [0,1]
    float3 voxelSize{1.f, 1.f, 1.f};
    float3 origin{0.f, 0.f, 0.f};
    float densityScale{1.f};
};

static inline bool hostIsLittleEndian()
{
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t *>(&x) == 1;
}

static inline uint16_t bswap16(uint16_t v) { return (uint16_t)((v >> 8) | (v << 8)); }

static bool parseRawArgs(int argc, char **argv, RawOptions &opt)
{
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        auto need = [&](int n)
        { return i + n < argc; };

        if (a == "-raw" && need(1))
        {
            opt.path = argv[++i];
        }
        else if (a == "-dim" && need(3))
        {
            opt.dim = make_int3(std::stoi(argv[++i]), std::stoi(argv[++i]), std::stoi(argv[++i]));
        }
        else if (a == "-bpp" && need(1))
        {
            opt.bpp = std::stoi(argv[++i]);
        }
        else if (a == "-le" && need(1))
        {
            opt.littleEndian = std::stoi(argv[++i]) != 0;
        }
        else if (a == "-norm" && need(1))
        {
            opt.normalize = std::stoi(argv[++i]) != 0;
        }
        else if (a == "-vox" && need(3))
        {
            opt.voxelSize = make_float3(std::stof(argv[++i]), std::stof(argv[++i]), std::stof(argv[++i]));
        }
        else if (a == "-org" && need(3))
        {
            opt.origin = make_float3(std::stof(argv[++i]), std::stof(argv[++i]), std::stof(argv[++i]));
        }
        else if (a == "-dens" && need(1))
        {
            opt.densityScale = std::stof(argv[++i]);
        }
    }
    if (opt.path.empty())
        return false; // 没给 -raw 就用 dummy
    if (opt.dim.x <= 0 || opt.dim.y <= 0 || opt.dim.z <= 0)
    { // 给了 -raw 但没给维度 => 错
        std::cerr << "[RAW] Missing or invalid -dim WxHxD.\n";
        return false;
    }
    if (opt.bpp != 8 && opt.bpp != 16)
    {
        std::cerr << "[RAW] Only 8 or 16 bits-per-voxel supported.\n";
        return false;
    }
    return true;
}

static bool readAllBytes(const std::string &p, std::vector<uint8_t> &out)
{
    std::ifstream ifs(p, std::ios::binary | std::ios::ate);
    if (!ifs)
        return false;
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize((size_t)size);
    return (bool)ifs.read((char *)out.data(), size);
}

// 实际构建 Volume：把 8/16 位标量转 float，并设置描述信息
std::shared_ptr<Volume> loadRawVolume(const RawOptions &opt);
