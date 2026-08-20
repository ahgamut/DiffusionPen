#!/usr/bin/env bash
# Build the per-writer style bank [W, feat] -> $STYLE_BANK (stage 3), over the
# merged split at $DATA_DIR. Pass --style-name to match the diffusion/style
# encoder (e.g. `scripts/local/build_style_bank.sh --style-name resnet18`); it
# must agree with the checkpoint you'll use --style-bank with, or the style dims
# won't match.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY -m utils.build_style_bank \
  --device "$DEVICE" \
  --data-dir "$DATA_DIR" \
  --style-path "$STYLE_PATH" \
  --out "$STYLE_BANK" \
  "$@"
