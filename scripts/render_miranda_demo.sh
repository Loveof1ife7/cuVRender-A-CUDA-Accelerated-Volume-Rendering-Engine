#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/nvcc-release"

VTI_PATH="${VTI_PATH:-/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat/miranda_1024x1024x1024_float32_part_0000/_canonical.vti}"
DATASET_DIR="${DATASET_DIR:-/root/autodl-tmp/projects/ff_w2gs/train-datas/high_quality_sim/miranda_1024x1024x1024_float32_part_0000}"
TF_INDEX="${TF_INDEX:-1}"
FRAME="${FRAME:-0}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
STEP="${STEP:-0.005}"
OPACITY="${OPACITY:-0.2}"
DENSITY="${DENSITY:-3.0}"
OUTPUT="${OUTPUT:-${BUILD_DIR}/miranda_demo_tf${TF_INDEX}_frame${FRAME}.png}"

TF_DIR="${DATASET_DIR}/miranda_1024x1024x1024_float32_part_0000_${TF_INDEX}"

cmake --build "${BUILD_DIR}" --target cuda-volume-renderer-cli -j 4

"${BUILD_DIR}/cuda-volume-renderer-cli" \
  --vti "${VTI_PATH}" \
  --tf "${TF_DIR}/tf_config.json" \
  --transforms "${TF_DIR}/transforms_train.json" \
  --frame "${FRAME}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --step "${STEP}" \
  --opacity "${OPACITY}" \
  --density "${DENSITY}" \
  --output "${OUTPUT}"

echo "Wrote ${OUTPUT}"
