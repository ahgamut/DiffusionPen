#!/usr/bin/env bash
# Build the per-writer style bank [339,1280] -> $STYLE_BANK (stage 3).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY -m utils.build_style_bank \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --style-path "$STYLE_PATH" \
  --out "$STYLE_BANK" \
  "$@"
