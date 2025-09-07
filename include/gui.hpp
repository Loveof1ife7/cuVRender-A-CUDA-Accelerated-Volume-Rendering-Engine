#include <vector> // for std::vector
#include <string> // for std::string

struct GUIState
{
    // 渲染参数
    float step_size = 0.5f;
    float opacity_scale = 1.0f;
    int render_mode = 0;
    float iso_value = 0.5f;

    // 相机参数
    float camera_pos[3] = {-50.0f, -50.0f, 0.0f};
    float camera_target[3] = {0.0f, 0.0f, 0.0f};
    float fov = 45.0f;

    // 光照参数
    float light_pos[3] = {2.0f, 2.0f, 2.0f};
    float light_color[3] = {1.0f, 1.0f, 1.0f};
    float light_intensity = 1.0f;

    // 传输函数参数
    float tf_points[8] = {0.0f, 0.0f, 0.3f, 0.5f, 0.6f, 0.8f, 1.0f, 1.0f};                          // alpha值
    float tf_colors[24] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1}; // RGB

    // 裁剪框参数
    float clip_min[3] = {-1000.0f, -1000.0f, -1000.0f};
    float clip_max[3] = {1000.0f, 1000.0f, 1000.0f};

    // 体积数据参数
    char volume_path[256] = "";
    int volume_dims[3] = {256, 256, 256};
    float volume_spacing[3] = {1.0f, 1.0f, 1.0f};
    int data_type = 0; // 0: uint8, 1: uint16, 2: float
    bool volume_loaded = false;

    // 文件浏览器
    bool show_file_browser = false;
    std::vector<std::string> available_files;
};
class Gui
{
};