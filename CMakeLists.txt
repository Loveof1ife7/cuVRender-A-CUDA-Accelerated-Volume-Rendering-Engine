cmake_minimum_required(VERSION 3.8.0)

project(cuda-volume-renderer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler=-O3")
set(CMAKE_CUDA_ARCHITECTURES "86")

enable_language(CUDA)
find_package(ZLIB REQUIRED)

option(CVR_BUILD_GUI "Build the GLFW/OpenGL demo executable when dependencies are available." ON)

set(CVR_CORE_SOURCES
    src/camera.cu
    src/cuda_utils.cu
    src/debug_utils.cu
    src/device_structs.cu
    src/io.cu
    src/lights.cu
    src/renderer_kernerl.cu
    src/scene.cu
    src/transfer_function.cu
    src/volume.cu
)

add_library(cvr_core STATIC ${CVR_CORE_SOURCES})
set_target_properties(cvr_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 86
)
target_include_directories(cvr_core
    PUBLIC
        include
        src
        third-party
)
target_compile_options(cvr_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcudafe "--diag_suppress=20236"
    >
)
target_link_libraries(cvr_core PUBLIC ${CMAKE_DL_LIBS} ZLIB::ZLIB)

add_executable(cuda-volume-renderer-cli src/headless_main.cu)
set_target_properties(cuda-volume-renderer-cli PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 86
)
target_include_directories(cuda-volume-renderer-cli PRIVATE include src third-party)
target_link_libraries(cuda-volume-renderer-cli PRIVATE cvr_core)

if(CVR_BUILD_GUI)
    cmake_policy(SET CMP0072 NEW)
    set(OpenGL_GL_PREFERENCE GLVND)

    find_package(OpenGL QUIET)
    find_package(glfw3 3.2 QUIET)

    if(OpenGL_FOUND AND glfw3_FOUND)
        message(STATUS "Building GUI demo target.")

        add_library(glad STATIC third-party/glad/src/glad.c)
        target_include_directories(glad PUBLIC third-party/glad/include)

        file(GLOB IMGUI_CORE_SOURCES
            "third-party/imgui/*.cpp"
        )
        set(IMGUI_BACKEND_SOURCES
            third-party/imgui/backends/imgui_impl_glfw.cpp
            third-party/imgui/backends/imgui_impl_opengl3.cpp
        )
        add_library(imgui STATIC ${IMGUI_CORE_SOURCES} ${IMGUI_BACKEND_SOURCES})
        target_include_directories(imgui PUBLIC
            third-party/imgui
            third-party/imgui/backends
        )
        target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
        target_link_libraries(imgui PUBLIC glad glfw)

        add_executable(cuda-volume-renderer
            src/main.cu
            src/image_presenter.cpp
            src/volume_renderer.cu
        )
        set_target_properties(cuda-volume-renderer PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES 86
        )
        target_include_directories(cuda-volume-renderer PRIVATE
            include
            src
            third-party
            third-party/imgui
            third-party/imgui/backends
        )
        target_link_libraries(cuda-volume-renderer PRIVATE
            cvr_core
            glad
            glfw
            OpenGL::GL
            imgui
        )
    else()
        message(WARNING "OpenGL/GLFW development packages were not found. Skipping GUI target and keeping the headless CLI build.")
    endif()
endif()

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

if(RES_FONT)
    foreach(file IN LISTS RES_FONT)
        file(COPY ${file} DESTINATION ${RES_DIR}/fonts)
    endforeach()
endif()

target_compile_definitions(cvr_core PUBLIC RESOURCES_DIRECTORY="${RES_DIR}")
target_compile_definitions(cuda-volume-renderer-cli PRIVATE RESOURCES_DIRECTORY="${RES_DIR}")
if(TARGET cuda-volume-renderer)
    target_compile_definitions(cuda-volume-renderer PRIVATE RESOURCES_DIRECTORY="${RES_DIR}")
endif()

if(WIN32)
    target_compile_definitions(cvr_core PRIVATE _AMD64_)
    if(TARGET cuda-volume-renderer)
        target_compile_definitions(cuda-volume-renderer PRIVATE _AMD64_)
    endif()
endif()
