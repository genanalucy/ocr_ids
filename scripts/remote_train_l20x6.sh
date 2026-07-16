#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCR_IDS_REMOTE_PROJECT:-/home/hzh/ocr_ids}"
RUNTIME_ROOT="${OCR_IDS_REMOTE_RUNTIME:-/home/hzh/ocr_ids_runtime}"
GPU_IDS="${OCR_IDS_GPU_IDS:?Set OCR_IDS_GPU_IDS to six confirmed-free physical GPU IDs}"
NPROC="${OCR_IDS_NPROC:-6}"
TRAIN_CONFIG="${OCR_IDS_TRAIN_CONFIG:-configs/train_l20x6.yaml}"

cd "$PROJECT_ROOT"
source "$RUNTIME_ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

IFS=',' read -r -a SELECTED_GPUS <<< "$GPU_IDS"
if [[ "${#SELECTED_GPUS[@]}" -ne "$NPROC" ]]; then
  echo "GPU_IDS contains ${#SELECTED_GPUS[@]} devices, expected NPROC=$NPROC" >&2
  exit 2
fi

if [[ "${OCR_IDS_ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  for gpu_id in "${SELECTED_GPUS[@]}"; do
    used_mib="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( used_mib > 1024 )); then
      echo "GPU $gpu_id is using ${used_mib} MiB; refusing to start." >&2
      exit 3
    fi
  done
fi

.venv/bin/torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  scripts/train.py --config "$TRAIN_CONFIG"
