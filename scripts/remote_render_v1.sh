#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCR_IDS_REMOTE_PROJECT:-/home/hzh/ocr_ids}"
RUNTIME_ROOT="${OCR_IDS_REMOTE_RUNTIME:-/home/hzh/ocr_ids_runtime}"
source "$RUNTIME_ROOT/env.sh"
cd "$PROJECT_ROOT"

FONT_ROOT="$OCR_IDS_DATA_ROOT/raw/fonts/noto-cjk/extracted/usr/share/fonts/opentype/noto"
SOURCE_ROOT="$OCR_IDS_DATA_ROOT/processed/splits-v1"
TARGET_ROOT="$OCR_IDS_DATA_ROOT/processed/rendered-v1"
mkdir -p "$TARGET_ROOT"

FONTS=(
  "$FONT_ROOT/NotoSansCJK-Regular.ttc#0" "$FONT_ROOT/NotoSansCJK-Regular.ttc#1"
  "$FONT_ROOT/NotoSansCJK-Regular.ttc#2" "$FONT_ROOT/NotoSansCJK-Regular.ttc#3"
  "$FONT_ROOT/NotoSansCJK-Regular.ttc#4" "$FONT_ROOT/NotoSansCJK-Bold.ttc#0"
  "$FONT_ROOT/NotoSansCJK-Bold.ttc#1" "$FONT_ROOT/NotoSansCJK-Bold.ttc#2"
  "$FONT_ROOT/NotoSansCJK-Bold.ttc#3" "$FONT_ROOT/NotoSansCJK-Bold.ttc#4"
  "$FONT_ROOT/NotoSerifCJK-Regular.ttc#0" "$FONT_ROOT/NotoSerifCJK-Regular.ttc#1"
  "$FONT_ROOT/NotoSerifCJK-Regular.ttc#2" "$FONT_ROOT/NotoSerifCJK-Regular.ttc#3"
  "$FONT_ROOT/NotoSerifCJK-Bold.ttc#0" "$FONT_ROOT/NotoSerifCJK-Bold.ttc#1"
  "$FONT_ROOT/NotoSerifCJK-Bold.ttc#2" "$FONT_ROOT/NotoSerifCJK-Bold.ttc#3"
)

font_args=()
for font in "${FONTS[@]}"; do
  font_args+=(--font "$font")
done

for split in train validation test_zero_char; do
  output="$TARGET_ROOT/$split"
  if [[ -f "$output/manifest.jsonl" ]]; then
    echo "Skipping completed $split: $output/manifest.jsonl"
    continue
  fi
  rm -rf "$output"
  .venv/bin/python scripts/render_dataset.py \
    --labels "$SOURCE_ROOT/$split.jsonl" \
    "${font_args[@]}" \
    --output "$output" --size 224 --font-size 184
done

.venv/bin/python scripts/holdout_font_samples.py \
  "$TARGET_ROOT/train/manifest.jsonl" \
  --train-output "$TARGET_ROOT/train/train.jsonl" \
  --test-output "$TARGET_ROOT/train/test_closed.jsonl"

echo "Rendering complete: $TARGET_ROOT"

