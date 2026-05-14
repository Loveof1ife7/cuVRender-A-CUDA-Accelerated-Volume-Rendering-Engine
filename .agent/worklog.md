# 2026-05-14

## Objective

- Test, unblock, and optimize the CUDA volume renderer so it can be built and executed in the current environment.
- Keep a reproducible log of changes, failures, fixes, and verification artifacts.

## Environment Findings

- Repository path: `/root/autodl-tmp/projects/cuda-accelerated-volume-renderer`
- Toolchain detected:
  - `cmake 3.22.1`
  - `ninja 1.13.0`
  - `nvcc 12.1.105`
- Current environment does **not** provide OpenGL development packages required by the original GLFW/GLAD/ImGui demo target.

## Initial Blockers

1. `cmake --preset nvcc-release` failed because `FindOpenGL` could not locate the OpenGL headers/libraries.
2. The project was tightly coupled to OpenGL:
   - `main.cu`
   - `image_presenter.cpp`
   - `volume_renderer.cu`
   - `debug_utils.*`
3. Several CUDA-side correctness/stability issues existed:
   - `Scene::commit()` never marked the device snapshot dirty, so `snapshotDevice()` could remain stale.
   - `Lights::set()` did not refresh an existing device buffer when contents changed.
   - Multiple `cudaMalloc()` calls used typed pointers without explicit `void**` casts.
   - `VolumeRenderer` destroyed the GL texture before unregistering CUDA interop.
   - `mode=1` and `mode=2` paths in the render kernel were incomplete.

## Changes Made

### Build System

- Refactored `CMakeLists.txt` into:
  - `cvr_core` static library for CUDA/core logic.
  - `cuda-volume-renderer-cli` headless executable that always builds.
  - `cuda-volume-renderer` GUI executable that is now optional and only built when OpenGL + GLFW are found.
- Added graceful fallback messaging when GUI dependencies are unavailable.
- Kept resource copying active for both CLI and GUI targets.

### Headless Demo Path

- Added `src/headless_main.cu`.
- Implemented a server-friendly CLI render path that:
  - builds a scene,
  - launches the CUDA render kernel directly,
  - copies the framebuffer back to host memory,
  - writes PNG output through `stb_image_write`.
- Added CLI parameters:
  - `--width`
  - `--height`
  - `--step`
  - `--mode`
  - `--output`
  - existing `-raw/-dim/-bpp/...` RAW loading path remains supported.

### Core Code Fixes

- Removed OpenGL-only helper code from `debug_utils.*` so the core library compiles headlessly.
- Fixed `Scene::commit()` dirty propagation and initial device upload behavior.
- Fixed `Lights::set()` to reallocate/copy correctly when the light list changes.
- Fixed CUDA allocation calls to use proper casts.
- Added safer inverse voxel handling in `worldToUVW()`.
- Implemented basic output behavior for:
  - `mode=1` ISO
  - `mode=2` MIP
- Hardened CUDA/GL interop calls in `volume_renderer.cu` with `CUDA_CHECK(...)`.

## Validation Performed

### Configure

- Command:
  - `cmake --preset nvcc-release`
- Result:
  - success
  - GUI target skipped as expected because OpenGL/GLFW dev packages are missing in this environment.

### Build

- Command:
  - `cmake --build build/nvcc-release --target cuda-volume-renderer-cli -j 4`
- Result:
  - success

### Runtime Smoke Tests

1. Dummy volume DVR:
   - Command:
     - `./cuda-volume-renderer-cli --width 640 --height 480 --output smoke.png`
   - Result:
     - success
     - produced `/root/autodl-tmp/projects/cuda-accelerated-volume-renderer/build/nvcc-release/smoke.png`

2. Dummy volume ISO:
   - Command:
     - `./cuda-volume-renderer-cli --width 640 --height 480 --mode 1 --output smoke-iso.png`
   - Result:
     - success

3. Dummy volume MIP:
   - Command:
     - `./cuda-volume-renderer-cli --width 640 --height 480 --mode 2 --output smoke-mip.png`
   - Result:
     - success

4. Synthetic RAW input path:
   - Generated a `32x32x32` synthetic 8-bit RAW sphere volume.
   - Command:
     - `./cuda-volume-renderer-cli -raw synthetic_32.raw -dim 32 32 32 --width 512 --height 512 --output synthetic-raw.png`
   - Result:
     - success
     - RAW loader reported expected dimensions/range and produced a valid PNG.

## Artifacts Generated

- `build/nvcc-release/smoke.png`
- `build/nvcc-release/smoke-iso.png`
- `build/nvcc-release/smoke-mip.png`
- `build/nvcc-release/synthetic_32.raw`
- `build/nvcc-release/synthetic-raw.png`

## Remaining Gaps / Next Steps

1. GUI demo target still requires system OpenGL/GLFW development packages before it can be compiled in this environment.
2. The built-in volume files under `resources/*.bin` are present, but metadata describing their dimensions/bit-depth is not yet wired into the CLI, so they were not used for this round of validation.
3. ISO mode is now functional at a basic level, but still lacks gradient-based shading and more polished surface classification.
4. Performance optimization is still at the “make it robust and runnable” stage; the next pass should target:
   - adaptive step sizing,
   - empty-space skipping,
   - gradient precomputation / shading path,
   - transfer-function tuning,
   - benchmark scripts for repeatable timing.

# 2026-05-14 - Miranda Demo Integration

## Objective

- Move from smoke-test rendering to a professional demo path using the Miranda dataset bundle:
  - canonical VTI volume,
  - Vol2Splat `tf_config.json`,
  - `transforms_train.json` camera trajectories,
  - reference render/QC metadata.

## Dataset Findings

- TF/camera/reference bundle:
  - `/root/autodl-tmp/projects/ff_w2gs/train-datas/high_quality_sim/miranda_1024x1024x1024_float32_part_0000`
- Canonical volume found at:
  - `/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat/miranda_1024x1024x1024_float32_part_0000/_canonical.vti`
- `_canonical.vti` is:
  - VTK ImageData,
  - `Float32`,
  - zlib-compressed appended base64,
  - `512x512x512`,
  - scalar range `[0, 1]`.
- The transforms JSON uses a render-space transform:
  - original bounds: `[0, 511]^3`
  - render bounds: `[-1.3, 1.3]^3`
  - scale factor: `0.0050880626`
  - offset: `[-1.3, -1.3, -1.3]`

## Changes Made

- Added zlib dependency to CMake for VTI decompression.
- Added VTI loader support:
  - parses VTK ImageData metadata,
  - decodes VTK block-compressed base64 correctly,
  - decompresses directly into the float scalar buffer to reduce peak memory.
- Added `tf_config.json` support:
  - parses Vol2Splat control points,
  - resamples them into a CUDA 1D transfer-function texture.
- Added `transforms_train/test.json` camera support:
  - extracts resolution and FOV,
  - converts OpenCV-style camera-to-world matrices into the renderer camera frame,
  - applies `render_world_transform` to the VTI volume origin/spacing.
- Added CLI options:
  - `--vti`
  - `--tf`
  - `--transforms`
  - `--frame`
  - `--opacity`
  - `--density`
  - `--iso`
  - `--tf-samples`
- Added lightweight gradient-based shading for DVR and ISO rendering.
- Added `scripts/render_miranda_demo.sh` for repeatable demo rendering.

## Validation Performed

- Rebuilt:
  - `cmake --build build/nvcc-release --target cuda-volume-renderer-cli -j 4`
- Smoke-tested dummy volume after input changes:
  - `./cuda-volume-renderer-cli --width 256 --height 192 --output smoke-after-vti.png`
- Smoke-tested TF loading:
  - `./cuda-volume-renderer-cli --tf .../tf_config.json --width 256 --height 192 --output smoke-tf.png`
- Rendered Miranda TF1 frame 0:
  - `./cuda-volume-renderer-cli --vti .../_canonical.vti --tf .../tf_config.json --transforms .../transforms_train.json --frame 0 --step 0.005 --opacity 0.2 --density 3.0 --output miranda-tf1-frame0-shaded.png`
- Rendered Miranda TF1 frame 1 at 256x256:
  - `./cuda-volume-renderer-cli --vti .../_canonical.vti --tf .../tf_config.json --transforms .../transforms_train.json --frame 1 --width 256 --height 256 --step 0.006 --opacity 0.2 --density 3.0 --output miranda-tf1-frame1-256.png`

## Artifacts Generated

- `build/nvcc-release/miranda-tf1-frame0.png`
- `build/nvcc-release/miranda-tf1-frame0-shaded.png`
- `build/nvcc-release/miranda-tf1-frame1-256.png`
- `build/nvcc-release/smoke-after-vti.png`
- `build/nvcc-release/smoke-tf.png`

## Remaining Gaps / Next Steps

- VTI loading is correct but still file-buffer heavy because the XML/appended payload is read as one string. It is usable for the current 512^3 canonical Miranda volume; a streaming base64 reader would be better for larger volumes.
- Current shading is finite-difference based and improves demo structure, but it costs extra texture samples. A precomputed gradient texture or mode switch would be the next performance pass.
- The renderer now consumes Miranda TF/camera data, but it does not yet batch-render all eight TFs and QC them against the provided reference metrics.
