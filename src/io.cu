#include "io.hpp"

std::shared_ptr<Volume> loadRawVolume(const RawOptions &opt)
{
    const size_t voxelCount = (size_t)opt.dim.x * opt.dim.y * opt.dim.z;
    const size_t expectedBytes = voxelCount * (opt.bpp / 8);

    std::vector<uint8_t> bytes;
    if (!readAllBytes(opt.path, bytes))
    {
        std::cerr << "[RAW] Failed to read file: " << opt.path << "\n";
        return nullptr;
    }
    if (bytes.size() != expectedBytes)
    {
        std::cerr << "[RAW] File size mismatch. Expect " << expectedBytes
                  << " bytes, got " << bytes.size() << " bytes.\n";
        return nullptr;
    }

    std::vector<float> scalars(voxelCount);
    float vmin = std::numeric_limits<float>::max();
    float vmax = std::numeric_limits<float>::lowest();

    if (opt.bpp == 8)
    {
        for (size_t i = 0; i < voxelCount; ++i)
        {
            uint8_t v = bytes[i];
            float f = opt.normalize ? (float)v / 255.f : (float)v;
            scalars[i] = f;
            vmin = std::min(vmin, f);
            vmax = std::max(vmax, f);
        }
    }
    else
    { // 16-bit
        const bool hostLE = hostIsLittleEndian();
        const uint16_t *src = reinterpret_cast<const uint16_t *>(bytes.data());
        for (size_t i = 0; i < voxelCount; ++i)
        {
            uint16_t w = src[i];
            if (hostLE != opt.littleEndian)
                w = bswap16(w);
            float f = opt.normalize ? (float)w / 65535.f : (float)w;
            scalars[i] = f;
            vmin = std::min(vmin, f);
            vmax = std::max(vmax, f);
        }
    }

    Volume::Description desc;
    desc.dim = opt.dim;
    desc.origin = opt.origin;
    desc.voxelSize = opt.voxelSize;
    desc.valueRange = make_float2(vmin, vmax); // 若 normalize==true，通常是 [0,1]
    desc.densityScale = opt.densityScale;

    std::cout << "[RAW] Loaded " << opt.path << " "
              << desc.dim.x << "x" << desc.dim.y << "x" << desc.dim.z
              << " bpp=" << opt.bpp
              << " range=[" << vmin << ", " << vmax << "]"
              << " normalize=" << (opt.normalize ? "1" : "0") << "\n";

    return std::make_shared<Volume>(desc, scalars.data());
}