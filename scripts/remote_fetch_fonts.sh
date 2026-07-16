#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${OCR_IDS_REMOTE_RUNTIME:-/home/hzh/ocr_ids_runtime}"
if [[ -f "$RUNTIME_ROOT/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ROOT/env.sh"
fi
DATA_ROOT="${OCR_IDS_DATA_ROOT:-$RUNTIME_ROOT/datasets}"
FONT_ROOT="$DATA_ROOT/raw/fonts/noto-cjk"
PACKAGE_ROOT="$FONT_ROOT/package"
EXTRACT_ROOT="$FONT_ROOT/extracted"
mkdir -p "$PACKAGE_ROOT" "$EXTRACT_ROOT"

if ! compgen -G "$PACKAGE_ROOT/fonts-noto-cjk_*.deb" > /dev/null; then
  (cd "$PACKAGE_ROOT" && apt download fonts-noto-cjk)
fi
DEB_FILE="$(find "$PACKAGE_ROOT" -maxdepth 1 -name 'fonts-noto-cjk_*.deb' -print -quit)"
rm -rf "$EXTRACT_ROOT"
dpkg-deb --extract "$DEB_FILE" "$EXTRACT_ROOT"

find "$EXTRACT_ROOT/usr/share/fonts" -type f \
  \( -name '*.otf' -o -name '*.ttf' -o -name '*.ttc' \) -print0 \
  | sort -z | xargs -0 sha256sum > "$FONT_ROOT/SHA256SUMS"
PACKAGE_VERSION="$(dpkg-deb --field "$DEB_FILE" Version)"
printf '%s\t%s\t%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "Ubuntu fonts-noto-cjk" "$PACKAGE_VERSION" \
  > "$FONT_ROOT/PROVENANCE.tsv"

echo "Fonts are stored only on the remote host under $FONT_ROOT"
find "$EXTRACT_ROOT/usr/share/fonts" -type f \
  \( -name '*.otf' -o -name '*.ttf' -o -name '*.ttc' \) -print

