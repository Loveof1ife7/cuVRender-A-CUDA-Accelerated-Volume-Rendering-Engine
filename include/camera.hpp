#pragma once
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "ray.hpp"
#include "device_structs.hpp"

//! host-only, package to to DeviceCamera
class Camera
{
public:
    Camera(const Eigen::Vector3f &pos,
           float vertical_fov_degrees,
           const Eigen::Vector2i &film_res);

    //* Set camera orientation by look-at target
    void lookAt(const Eigen::Vector3f &target,
                const Eigen::Vector3f &ref_up);

    void moveTo(const Eigen::Vector3f &pos);

    //* Generate ray through pixel (dx, dy)
    Ray generateRay(float dx, float dy) const;

    //* Getter
    Eigen::Vector3f getPosition() const { return position_; }
    Eigen::Vector3f getForward() const { return forward_; }
    Eigen::Vector3f getUp() const { return up_; }
    Eigen::Vector3f getRight() const { return right_; }
    float getVerticalFov() const { return vertical_fov_deg_; }
    Eigen::Vector2i getFilmResolution() const { return film_res_; }

    //* Setter
    void SetFilmResolution(Eigen::Vector2i res) { film_res_ = res; }
    void SetVerticalFov(float degree) { vertical_fov_deg_ = degree; }

    //* Convert to device representation
    DeviceCamera toDevice() const;

private:
    void orthonormalize();

    //* host data
    Eigen::Vector3f position_{0,
                              0,
                              0};
    Eigen::Vector3f forward_{0, 0, -1};
    Eigen::Vector3f up_{0, 1, 0};
    Eigen::Vector3f right_{1, 0, 0};
    float vertical_fov_deg_ = 45.f;
    Eigen::Vector2i film_res_{1, 1};

    //* device data
    DeviceCamera *d_camera;
};