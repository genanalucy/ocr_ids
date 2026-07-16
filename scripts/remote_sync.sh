#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/remote.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/remote.env"
fi

REMOTE_HOST="${OCR_IDS_REMOTE_HOST:-hzh@10.240.147.134}"
REMOTE_PROJECT="${OCR_IDS_REMOTE_PROJECT:-/home/hzh/ocr_ids}"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'data/raw/' \
  --exclude 'data/interim/' \
  --exclude 'data/processed/' \
  --exclude 'data/images/' \
  --exclude 'checkpoints/' \
  --exclude 'outputs/' \
  --exclude 'runs/' \
  --exclude 'remote.env' \
  "$ROOT/" "$REMOTE_HOST:$REMOTE_PROJECT/"

echo "Synced code only to $REMOTE_HOST:$REMOTE_PROJECT"

