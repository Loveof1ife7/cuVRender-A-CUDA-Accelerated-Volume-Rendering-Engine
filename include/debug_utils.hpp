#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <memory>
#include <vector>
#include <cmath>
#include <cstdio>
#include "volume.hpp"

#include "transfer_function.hpp"

GLuint createGradientTexture(int w, int h);

GLuint makeChecker(int w, int h);

std::shared_ptr<Volume> makeDummyVolume();

std::shared_ptr<TransferFunction> makeSimpleTF();

__global__ void kernelCheckCUDAGL(DeviceScene ds, uchar4 *out, int w, int h);

__global__ void kernelSampleCenter(DeviceScene ds, uchar4 *out, int w, int h);

__global__ void kernelSliceUV(DeviceScene ds, uchar4 *out, int w, int h);

__global__ void kernelRayHit(DeviceScene ds, uchar4 *out, int w, int h);

__device__ void volumeAABB(const DeviceVolume &vol, float3 &bmin, float3 &bmax);

__device__ bool rayBox(const float3 ro, const float3 rd,
                       const float3 bmin, const float3 bmax,
                       float &tnear, float &tfar);