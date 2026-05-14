#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>

#include "image_presenter.hpp"
#include "scene.hpp"
#include "volume_renderer.hpp"
#include "cuda_utils.hpp"
#include "debug_utils.hpp"
#include "io.hpp"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

GLFWwindow *initWindow()
{
  if (!glfwInit())
  {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return nullptr;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "ImGui OpenGL Example", nullptr, nullptr);
  if (!window)
  {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return nullptr;
  }

  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return nullptr;
  }

  return window;
}

void initImGui(GLFWwindow *window)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;

  ImGui::StyleColorsDark();

  const char *glsl_version = "#version 100";
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  /* Load Fonts */
  // io.Fonts->AddFontFromFileTTF(
  //     (resource_manager.getResource("fonts/Roboto-Medium.ttf")).c_str(), 16.0f);
  // io.Fonts->AddFontDefault();
}

void shutdownImGui()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void renderScene(GLuint texID)
{
  glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
  glClear(GL_COLOR_BUFFER_BIT);

  glGenTextures(1, &texID);
}

void renderImGui()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Hello, SciVis!");
  ImGui::Text("Cuda-SciVis.");
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main(int argc, char **argv)
{

#ifdef DEBUG
  std::cout << "Debug mode is active!" << std::endl;
#endif

  GLFWwindow *window = initWindow();
  if (!window)
  {
    fprintf(stderr, "Failed to create GLFW window!\n");
    return 1;
  }

  initImGui(window);
  CUDA_CHECK(cudaSetDevice(0));

  ImagePresenter *presenter = new ImagePresenter(WINDOW_WIDTH, WINDOW_HEIGHT);

  VolumeRenderer::CreateInfo ci{WINDOW_WIDTH, WINDOW_HEIGHT, false};
  VolumeRenderer *renderer = new VolumeRenderer(ci);

  RawOptions opt;
  bool useRaw = parseRawArgs(argc, argv, opt);
  std::shared_ptr<Volume> volume;
  if (useRaw)
  {
    volume = loadRawVolume(opt);
    if (!volume)
    {
      std::cerr << "[RAW] Fallback to dummy volume.\n";
      volume = makeDummyVolume();
    }
  }
  else
  {
    volume = makeDummyVolume();
  }
  auto volume_center = volume->getVolumeCenter();
  auto tf = makeSimpleTF();

  Camera cam(Eigen::Vector3f(-50, -50, 0), /*fov*/ 45.0f, Eigen::Vector2i(WINDOW_WIDTH, WINDOW_HEIGHT));
  cam.lookAt(Eigen::Vector3f(volume_center.x, volume_center.y, volume_center.z), Eigen::Vector3f(0, 1, 0));

  Lights lights;
  std::vector<DeviceLight> ls(1);
  ls[0].position = make_float3(2, 2, 2);
  ls[0].color = make_float3(1, 1, 1);
  ls[0].intensity = 1.0f;
  ls[0].type = 0;
  lights.set(ls);

  Scene scene;
  scene.setCamera(&cam)
      .setLights(&lights)
      .setVolume(volume)
      .setTransferFunction(tf)
      .setRenderParams(/*step*/ 0.5f, /*opacityScale*/ 1.0f, /*mode*/ 0, /*iso*/ 0.5f)
      .setClipBox(make_float3(-1e3f, -1e3f, -1e3f), make_float3(1e3f, 1e3f, 1e3f));
  scene.commit(0);

  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    renderer->render(scene);

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glClearColor(0.1f, 0.1f, 0.1f, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // presenter->display(dbgTex);
    presenter->display(renderer->getResultTexture());

    glfwSwapBuffers(window);
  }

  shutdownImGui();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
