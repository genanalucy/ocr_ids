#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCR_IDS_REMOTE_PROJECT:-/home/hzh/ocr_ids}"
RUNTIME_ROOT="${OCR_IDS_REMOTE_RUNTIME:-/home/hzh/ocr_ids_runtime}"

mkdir -p \
  "$RUNTIME_ROOT/datasets/raw" \
  "$RUNTIME_ROOT/datasets/interim" \
  "$RUNTIME_ROOT/datasets/processed" \
  "$RUNTIME_ROOT/models" \
  "$RUNTIME_ROOT/runs" \
  "$RUNTIME_ROOT/cache/huggingface" \
  "$RUNTIME_ROOT/cache/torch"

cd "$PROJECT_ROOT"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
  echo "未找到 uv：$UV_BIN；请通过 UV_BIN 指定其路径" >&2
  exit 1
fi
"$UV_BIN" venv --python /usr/local/bin/python3.12 .venv
"$UV_BIN" pip install --python .venv/bin/python \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$UV_BIN" pip install --python .venv/bin/python -e '.[train,dev]'

cat > "$RUNTIME_ROOT/env.sh" <<EOF
export OCR_IDS_DATA_ROOT="$RUNTIME_ROOT/datasets"
export OCR_IDS_MODEL_ROOT="$RUNTIME_ROOT/models"
export OCR_IDS_RUNS_ROOT="$RUNTIME_ROOT/runs"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TORCH_HOME="$RUNTIME_ROOT/cache/torch"
EOF

echo "Remote environment ready. Run: source $RUNTIME_ROOT/env.sh"
