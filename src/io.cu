#include "io.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <zlib.h>

namespace
{
uint32_t readLE32(const std::vector<uint8_t> &bytes, size_t offset)
{
    if (offset + 4 > bytes.size())
        throw std::runtime_error("Unexpected end of VTI block header");
    return uint32_t(bytes[offset]) |
           (uint32_t(bytes[offset + 1]) << 8) |
           (uint32_t(bytes[offset + 2]) << 16) |
           (uint32_t(bytes[offset + 3]) << 24);
}

int base64Value(char c)
{
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}

void skipBase64Whitespace(const std::string &src, size_t &pos, size_t end)
{
    while (pos < end && std::isspace(static_cast<unsigned char>(src[pos])))
        ++pos;
}

std::vector<uint8_t> decodeBase64Segment(const std::string &src,
                                          size_t &pos,
                                          size_t end,
                                          size_t decodedBytes)
{
    std::vector<uint8_t> out;
    out.reserve(decodedBytes);

    while (out.size() < decodedBytes)
    {
        int vals[4] = {0, 0, 0, 0};
        int pad = 0;
        for (int i = 0; i < 4; ++i)
        {
            skipBase64Whitespace(src, pos, end);
            if (pos >= end)
                throw std::runtime_error("Unexpected end of VTI base64 payload");

            const char c = src[pos++];
            if (c == '=')
            {
                vals[i] = 0;
                ++pad;
            }
            else
            {
                vals[i] = base64Value(c);
                if (vals[i] < 0)
                    throw std::runtime_error("Invalid character in VTI base64 payload");
            }
        }

        const uint32_t triple = (uint32_t(vals[0]) << 18) |
                                (uint32_t(vals[1]) << 12) |
                                (uint32_t(vals[2]) << 6) |
                                uint32_t(vals[3]);
        if (out.size() < decodedBytes && pad < 3)
            out.push_back(static_cast<uint8_t>((triple >> 16) & 0xff));
        if (out.size() < decodedBytes && pad < 2)
            out.push_back(static_cast<uint8_t>((triple >> 8) & 0xff));
        if (out.size() < decodedBytes && pad < 1)
            out.push_back(static_cast<uint8_t>(triple & 0xff));
    }

    return out;
}

std::string readTextFile(const std::string &path)
{
    std::vector<uint8_t> bytes;
    if (!readAllBytes(path, bytes))
        throw std::runtime_error("Failed to read file: " + path);
    return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

bool extractQuotedAttribute(const std::string &text,
                            const std::string &name,
                            std::string &out)
{
    const std::string needle = name + "=\"";
    const size_t pos = text.find(needle);
    if (pos == std::string::npos)
        return false;
    const size_t begin = pos + needle.size();
    const size_t end = text.find('"', begin);
    if (end == std::string::npos)
        return false;
    out = text.substr(begin, end - begin);
    return true;
}

std::vector<double> parseNumbers(const std::string &text)
{
    std::vector<double> values;
    const char *p = text.c_str();
    char *end = nullptr;
    while (*p)
    {
        const double v = std::strtod(p, &end);
        if (end != p)
        {
            values.push_back(v);
            p = end;
        }
        else
        {
            ++p;
        }
    }
    return values;
}

std::vector<double> parseArrayForKey(const std::string &json, const std::string &key)
{
    const std::string needle = "\"" + key + "\"";
    const size_t keyPos = json.find(needle);
    if (keyPos == std::string::npos)
        return {};
    const size_t arrayBegin = json.find('[', keyPos);
    if (arrayBegin == std::string::npos)
        return {};

    int depth = 0;
    for (size_t i = arrayBegin; i < json.size(); ++i)
    {
        if (json[i] == '[')
            ++depth;
        else if (json[i] == ']')
        {
            --depth;
            if (depth == 0)
                return parseNumbers(json.substr(arrayBegin, i - arrayBegin + 1));
        }
    }
    return {};
}

double parseNumberForKey(const std::string &json, const std::string &key, double fallback)
{
    const std::string needle = "\"" + key + "\"";
    const size_t keyPos = json.find(needle);
    if (keyPos == std::string::npos)
        return fallback;
    const size_t colon = json.find(':', keyPos);
    if (colon == std::string::npos)
        return fallback;
    const char *p = json.c_str() + colon + 1;
    char *end = nullptr;
    const double value = std::strtod(p, &end);
    return end == p ? fallback : value;
}

std::vector<double> parseNthTransformMatrix(const std::string &json, int frameIndex)
{
    size_t search = 0;
    for (int i = 0; i <= frameIndex; ++i)
    {
        search = json.find("\"transform_matrix\"", search);
        if (search == std::string::npos)
            return {};
        search += 18;
    }

    const size_t arrayBegin = json.find('[', search);
    if (arrayBegin == std::string::npos)
        return {};

    int depth = 0;
    for (size_t i = arrayBegin; i < json.size(); ++i)
    {
        if (json[i] == '[')
            ++depth;
        else if (json[i] == ']')
        {
            --depth;
            if (depth == 0)
                return parseNumbers(json.substr(arrayBegin, i - arrayBegin + 1));
        }
    }
    return {};
}

float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}
} // namespace

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

std::shared_ptr<Volume> loadVtiVolume(const VtiOptions &opt)
{
    std::vector<uint8_t> fileBytes;
    if (!readAllBytes(opt.path, fileBytes))
        throw std::runtime_error("[VTI] Failed to read file: " + opt.path);

    const std::string file(reinterpret_cast<const char *>(fileBytes.data()), fileBytes.size());
    const size_t appendedTag = file.find("<AppendedData");
    if (appendedTag == std::string::npos)
        throw std::runtime_error("[VTI] Missing AppendedData section");

    const std::string header = file.substr(0, appendedTag);
    if (header.find("compressor=\"vtkZLibDataCompressor\"") == std::string::npos)
        throw std::runtime_error("[VTI] Only vtkZLibDataCompressor VTI files are supported");
    if (header.find("type=\"Float32\"") == std::string::npos)
        throw std::runtime_error("[VTI] Only Float32 scalar VTI files are supported");

    std::string extentText, originText, spacingText;
    if (!extractQuotedAttribute(header, "WholeExtent", extentText))
        throw std::runtime_error("[VTI] Missing WholeExtent");
    extractQuotedAttribute(header, "Origin", originText);
    extractQuotedAttribute(header, "Spacing", spacingText);

    const auto extent = parseNumbers(extentText);
    if (extent.size() != 6)
        throw std::runtime_error("[VTI] Invalid WholeExtent");

    const int3 dim = make_int3(
        int(extent[1] - extent[0] + 1),
        int(extent[3] - extent[2] + 1),
        int(extent[5] - extent[4] + 1));
    const size_t voxelCount = size_t(dim.x) * dim.y * dim.z;
    const size_t expectedBytes = voxelCount * sizeof(float);

    auto originValues = parseNumbers(originText);
    auto spacingValues = parseNumbers(spacingText);
    if (originValues.size() != 3)
        originValues = {0.0, 0.0, 0.0};
    if (spacingValues.size() != 3)
        spacingValues = {1.0, 1.0, 1.0};

    size_t pos = file.find('_', appendedTag);
    const size_t appendedEnd = file.find("</AppendedData>", appendedTag);
    if (pos == std::string::npos || appendedEnd == std::string::npos)
        throw std::runtime_error("[VTI] Invalid AppendedData payload");
    ++pos;

    std::vector<uint8_t> firstHeader = decodeBase64Segment(file, pos, appendedEnd, 12);
    const uint32_t blockCount = readLE32(firstHeader, 0);
    const uint32_t blockSize = readLE32(firstHeader, 4);
    const uint32_t lastBlockSize = readLE32(firstHeader, 8);
    if (blockCount == 0 || blockSize == 0)
        throw std::runtime_error("[VTI] Invalid compressed block header");

    std::vector<uint8_t> remainingHeader = decodeBase64Segment(file, pos, appendedEnd, size_t(blockCount) * 4);
    std::vector<uint32_t> compressedSizes(blockCount);
    size_t compressedTotalBytes = 0;
    for (uint32_t i = 0; i < blockCount; ++i)
    {
        compressedSizes[i] = readLE32(remainingHeader, size_t(i) * 4);
        compressedTotalBytes += compressedSizes[i];
    }

    std::vector<float> scalars(voxelCount);
    uint8_t *scalarBytes = reinterpret_cast<uint8_t *>(scalars.data());
    std::vector<uint8_t> compressedPayload = decodeBase64Segment(file, pos, appendedEnd, compressedTotalBytes);
    size_t outOffset = 0;
    size_t compressedOffset = 0;
    for (uint32_t i = 0; i < blockCount; ++i)
    {
        const size_t expectedBlockBytes = (i == blockCount - 1 && lastBlockSize != 0)
                                              ? lastBlockSize
                                              : blockSize;
        if (outOffset + expectedBlockBytes > expectedBytes)
            throw std::runtime_error("[VTI] Decompressed data is larger than expected volume size");

        uLongf dstLen = static_cast<uLongf>(expectedBlockBytes);
        const int zret = uncompress(scalarBytes + outOffset,
                                    &dstLen,
                                    compressedPayload.data() + compressedOffset,
                                    static_cast<uLong>(compressedSizes[i]));
        if (zret != Z_OK || dstLen != expectedBlockBytes)
            throw std::runtime_error("[VTI] zlib failed while decompressing a volume block");
        outOffset += expectedBlockBytes;
        compressedOffset += compressedSizes[i];
    }

    if (outOffset != expectedBytes)
        throw std::runtime_error("[VTI] Decompressed byte count does not match volume dimensions");

    float vmin = std::numeric_limits<float>::max();
    float vmax = std::numeric_limits<float>::lowest();
    for (float v : scalars)
    {
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }
    if (opt.normalize && vmax > vmin)
    {
        const float invRange = 1.0f / (vmax - vmin);
        for (float &v : scalars)
            v = (v - vmin) * invRange;
        vmin = 0.0f;
        vmax = 1.0f;
    }

    Volume::Description desc;
    desc.dim = dim;
    desc.origin = make_float3(float(originValues[0]), float(originValues[1]), float(originValues[2]));
    desc.voxelSize = make_float3(float(spacingValues[0]), float(spacingValues[1]), float(spacingValues[2]));
    if (opt.overrideWorldTransform)
    {
        desc.origin = opt.worldOrigin;
        desc.voxelSize = make_float3(
            desc.voxelSize.x * opt.worldSpacingScale,
            desc.voxelSize.y * opt.worldSpacingScale,
            desc.voxelSize.z * opt.worldSpacingScale);
    }
    desc.valueRange = make_float2(vmin, vmax);
    desc.densityScale = opt.densityScale;

    std::cout << "[VTI] Loaded " << opt.path << " "
              << dim.x << "x" << dim.y << "x" << dim.z
              << " range=[" << vmin << ", " << vmax << "]"
              << " blocks=" << blockCount << "\n";

    return std::make_shared<Volume>(desc, scalars.data());
}

std::shared_ptr<TransferFunction> loadTransferFunctionConfig(const std::string &path, int samples)
{
    const std::string json = readTextFile(path);
    auto dataRange = parseArrayForKey(json, "data_range");
    auto controlValues = parseArrayForKey(json, "control_points");
    if (dataRange.size() < 2 || controlValues.size() < 10 || controlValues.size() % 5 != 0)
        throw std::runtime_error("[TF] Invalid tf_config.json: " + path);

    struct CP
    {
        float x, r, g, b, a;
    };

    std::vector<CP> cps;
    for (size_t i = 0; i + 4 < controlValues.size(); i += 5)
    {
        cps.push_back(CP{
            float(controlValues[i + 0]),
            float(controlValues[i + 1]),
            float(controlValues[i + 2]),
            float(controlValues[i + 3]),
            float(controlValues[i + 4])});
    }
    std::sort(cps.begin(), cps.end(), [](const CP &a, const CP &b)
              { return a.x < b.x; });

    samples = std::max(2, samples);
    std::vector<float4> table(samples);
    size_t seg = 0;
    for (int i = 0; i < samples; ++i)
    {
        const float x = i / float(samples - 1);
        while (seg + 1 < cps.size() && cps[seg + 1].x < x)
            ++seg;

        const CP &a = cps[seg];
        const CP &b = cps[std::min(seg + 1, cps.size() - 1)];
        const float denom = std::max(b.x - a.x, 1e-6f);
        const float t = std::min(std::max((x - a.x) / denom, 0.0f), 1.0f);
        table[i] = make_float4(
            lerp(a.r, b.r, t),
            lerp(a.g, b.g, t),
            lerp(a.b, b.b, t),
            lerp(a.a, b.a, t));
    }

    std::cout << "[TF] Loaded " << path
              << " cps=" << cps.size()
              << " samples=" << samples
              << " domain=[" << dataRange[0] << ", " << dataRange[1] << "]\n";
    return std::make_shared<TransferFunction>(
        table.data(),
        static_cast<int>(table.size()),
        make_float2(float(dataRange[0]), float(dataRange[1])));
}

TransformCameraFrame loadTransformCameraFrame(const std::string &path, int frameIndex)
{
    const std::string json = readTextFile(path);
    const auto matrix = parseNthTransformMatrix(json, std::max(frameIndex, 0));
    if (matrix.size() != 16)
        throw std::runtime_error("[CAM] Could not parse transform_matrix frame from " + path);

    TransformCameraFrame out;
    out.valid = true;
    out.width = int(parseNumberForKey(json, "w", 0.0));
    out.height = int(parseNumberForKey(json, "h", 0.0));
    const double flY = parseNumberForKey(json, "fl_y", 0.0);
    if (out.height > 0 && flY > 0.0)
        out.verticalFovDegrees = float(2.0 * std::atan(out.height / (2.0 * flY)) * 180.0 / M_PI);

    out.position = Eigen::Vector3f(float(matrix[3]), float(matrix[7]), float(matrix[11]));

    const Eigen::Vector3f right{float(matrix[0]), float(matrix[4]), float(matrix[8])};
    const Eigen::Vector3f down{float(matrix[1]), float(matrix[5]), float(matrix[9])};
    const Eigen::Vector3f forward{float(matrix[2]), float(matrix[6]), float(matrix[10])};
    (void)right;

    out.forward = forward.normalized();
    out.up = (-down).normalized();

    const double scale = parseNumberForKey(json, "scale_factor", 0.0);
    const auto offsets = parseArrayForKey(json, "offset");
    if (scale > 0.0 && offsets.size() >= 3)
    {
        out.hasVolumeWorldTransform = true;
        out.volumeWorldSpacingScale = float(scale);
        out.volumeWorldOrigin = make_float3(float(offsets[0]), float(offsets[1]), float(offsets[2]));
    }

    std::cout << "[CAM] Loaded " << path
              << " frame=" << frameIndex
              << " resolution=" << out.width << "x" << out.height
              << " vfov=" << out.verticalFovDegrees;
    if (out.hasVolumeWorldTransform)
    {
        std::cout << " volume_origin=("
                  << out.volumeWorldOrigin.x << ","
                  << out.volumeWorldOrigin.y << ","
                  << out.volumeWorldOrigin.z << ")"
                  << " scale=" << out.volumeWorldSpacingScale;
    }
    std::cout << "\n";
    return out;
}
