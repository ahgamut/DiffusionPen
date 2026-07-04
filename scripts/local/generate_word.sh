#!/usr/bin/env bash
# Generate a single word crop (generate.py).
# Env: WRITER_ID, SAMPLING_WORD, OUTPUT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY generate.py \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --save-path "$SAVE_PATH" \
  --style-path "$STYLE_PATH" \
  --stable-dif-path "$STABLE_DIF_PATH" \
  --style-bank-path "$STYLE_BANK" \
  -w "$WRITER_ID" \
  --sampling-word "$SAMPLING_WORD" \
  -o "$OUTPUT" \
  "$@"
