cmake_minimum_required(VERSION 3.8.0)

project(cuda-volume-renderer LANGUAGES C CXX CUDA)

# Use modern CMake target-based design
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler=-O3")
set(CMAKE_CUDA_ARCHITECTURES "86")

# === Dependencies ===
# CUDA
enable_language(CUDA)
#find_package(CUDA REQUIRED)  # 确保找到 CUDA
#include_directories(${CUDA_INCLUDE_DIRS})  # 显式添加 CUDA 头文件路径

# OpenGL

cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
	message(STATUS "OpenGL found.")
	include_directories(${OPENGL_INCLUDE_DIR})
endif()

# glfw
find_package(glfw3 3.2 REQUIRED)
if(glfw3_FOUND)
	message(STATUS "GLFW3 found.")
endif()

# imGui
file(GLOB_RECURSE IMGUI_SOURCES
    "third-party/imgui/*.cpp"
    "third-party/imgui/*.h"
)
add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC 
    third-party/imgui
    third-party/imgui/backends)

target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)

target_link_libraries(imgui PUBLIC
    glad
    glfw)

# === Include and Link Third-Party Libraries ===
add_library(glad STATIC third-party/glad/src/glad.c)
target_include_directories(glad PUBLIC third-party/glad/include)

# === Executable ===
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")  
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
add_executable(cuda-volume-renderer ${CUDA_SOURCES} ${CPP_SOURCES})

# CUDA settings
set_target_properties(cuda-volume-renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON
CUDA_ARCHITECTURES 86  CUDA_RESOLVE_DEVICE_SYMBOLS ON )

target_include_directories(cuda-volume-renderer PRIVATE include src third-party third-party/imgui third-party/imgui/backends ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

target_link_libraries(cuda-volume-renderer 
    PRIVATE    
    glad 
    glfw 
    OpenGL::GL
    imgui
    ${CMAKE_DL_LIBS}
    ${CUDA_INCLUDE_DIRS}
    )

# === Resource Handling ===
file(GLOB_RECURSE RES_IMG CONFIGURE_DEPENDS resources/*.{png,jpg})
file(GLOB_RECURSE RES_BIN CONFIGURE_DEPENDS resources/*.bin)
file(GLOB_RECURSE RES_FONT CONFIGURE_DEPENDS third-party/imgui/misc/fonts/*.ttf)

set(RES_DIR ${CMAKE_BINARY_DIR}/resources)
file(MAKE_DIRECTORY ${RES_DIR}/images ${RES_DIR}/fonts ${RES_DIR}/bin)

foreach(file IN LISTS RES_IMG)
    file(COPY ${file} DESTINATION ${RES_DIR}/images)
endforeach()

foreach(file IN LISTS RES_BIN)
    file(COPY ${file} DESTINATION ${RES_DIR}/bin)
endforeach()

foreach(file IN LISTS RES_FONT)
    file(COPY ${file} DESTINATION ${RES_DIR}/fonts)
endforeach()

target_compile_definitions(cuda-volume-renderer PRIVATE RESOURCES_DIRECTORY="${RES_DIR}")

target_compile_options(cuda-volume-renderer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcudafe "--diag_suppress=20236"  # 禁用该特定警告
    >
)
if(WIN32)
    target_compile_definitions(cuda-volume-renderer PRIVATE _AMD64_)
endif()
