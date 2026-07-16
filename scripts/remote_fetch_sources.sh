#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${OCR_IDS_REMOTE_RUNTIME:-/home/hzh/ocr_ids_runtime}"
if [[ -f "$RUNTIME_ROOT/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ROOT/env.sh"
fi
DATA_ROOT="${OCR_IDS_DATA_ROOT:-$RUNTIME_ROOT/datasets}"
RAW_ROOT="$DATA_ROOT/raw"
mkdir -p "$RAW_ROOT"

sync_repo() {
  local name="$1"
  local url="$2"
  local destination="$RAW_ROOT/$name"
  if [[ -d "$destination/.git" ]]; then
    git -C "$destination" pull --ff-only
  else
    git clone --depth 1 "$url" "$destination"
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$name" "$url" "$(git -C "$destination" rev-parse HEAD)" \
    >> "$RAW_ROOT/PROVENANCE.tsv"
}

sync_repo cjkvi-ids https://github.com/cjkvi/cjkvi-ids.git
sync_repo chise-ids https://gitlab.chise.org/CHISE/ids.git

echo "Sources are stored only on the remote host under $RAW_ROOT"

