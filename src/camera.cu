#include "camera.hpp"
#include "cuda_utils.hpp"

using MathUtils::Camera::f3;

Camera::Camera(const Eigen::Vector3f &pos,
               float vertical_fov_degrees,
               const Eigen::Vector2i &film_res)
    : position_(pos),
      vertical_fov_deg_(vertical_fov_degrees),
      film_res_(film_res)
{
    forward_ = Eigen::Vector3f(0, 0, -1);
    up_ = Eigen::Vector3f(0, 1, 0);
    right_ = forward_.cross(up_).normalized();
    orthonormalize();
}

void Camera::lookAt(const Eigen::Vector3f &target,
                    const Eigen::Vector3f &ref_up)
{
    //* 前向方向
    Eigen::Vector3f f = (target - position_).normalized();
    Eigen::Vector3f r = f.cross(ref_up).normalized();
    Eigen::Vector3f u = r.cross(f).normalized();

    forward_ = f;
    right_ = r;
    up_ = u;
    orthonormalize();
}

void Camera::moveTo(const Eigen::Vector3f &pos)
{
    position_ = pos;
}

Ray Camera::generateRay(float x, float y) const
{
    //* (x, y) from [0,film_width] × [0, film_height] to [-1, 1] × [-1, 1] , NDC
    float u = ((x + 0.5f) / film_res_.x()) * 2.0f - 1.0f;
    float v = ((y + 0.5f) / film_res_.y()) * 2.0f - 1.0f;
    //* flip y
    v = -v;

    float aspect_ratio = static_cast<float>(film_res_.x() / (float)film_res_.y());

    float vertical_fov_rad = vertical_fov_deg_ * (M_PI / 180.0f);

    float half_height = std::tan(vertical_fov_rad / 2.0f);
    float half_width = aspect_ratio * half_height;

    //* camera space
    //* The imaging plane (the screen) defaults to the z = -1 plane in camera space
    //* The ray from the camera origin (0,0,0) to the imaging plane (x', y', -1), defines the direction corresponding to the pixel (u,v).
    Eigen::Vector3f dir_cam(u * half_width, v * half_height, -1.0f);
    dir_cam.normalize();

    //* world space
    Eigen::Vector3f dir_world = right_ * dir_cam.x() + up_ * dir_cam.y() + forward_ * dir_cam.z();

    return Ray(position_, dir_world.normalized());
}

void Camera::orthonormalize()
{
    // Gram-Schmidt
    forward_.normalize();
    right_ = forward_.cross(up_);
    if (right_.squaredNorm() < 1e-12f)
    {
        Eigen::Vector3f alt_up = std::abs(forward_.y()) < 0.999f ? Eigen::Vector3f(0, 1, 0)
                                                                 : Eigen::Vector3f(1, 0, 0);
        right_ = forward_.cross(alt_up);
    }
    right_.normalize();
    up_ = right_.cross(forward_).normalized();
}

DeviceCamera Camera::toDevice() const
{
    DeviceCamera dc{};
    dc.position_ = f3(position_);
    dc.forward_ = f3(forward_.normalized());
    dc.up_ = f3(up_.normalized());
    dc.right_ = f3(right_.normalized());
    dc.vertical_fov_ = vertical_fov_deg_;
    return dc;
}

//! api usage
//* Scene::commit()
//* if(m_dirty_camera)
//* {
//*     m_ds_host.camera = m_cam->toDevice();
//*     m_dirty_camera = false;
//* }
