#pragma once
#include <Eigen/Dense>
#include <float.h>

struct Ray
{
    Eigen::Vector3f origin;
    Eigen::Vector3f direction;
    float tmin = 0.0f;
    float tmax = FLT_MAX;
    Ray(Eigen::Vector3f origin, Eigen::Vector3f direction, float tmin = 0.0f, float tmax = FLT_MAX)
        : origin(origin), direction(direction), tmin(tmin), tmax(tmax) {}
};