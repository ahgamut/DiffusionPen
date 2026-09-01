#!/usr/bin/env bash
# Generate a paragraph (generation/gen_paragraph.py).
# Env: WRITER_ID, PROMPT_FILE, OUTPUT, PLACEMENT=heuristic|learned, MAX_LINE_WIDTH.
# For learned layout / upscaling, PLACEMENT=learned and/or append --upsample:
#   PLACEMENT=learned scripts/local/generate_paragraph.sh --upsample --upsampler-path "$UPSAMPLER_CKPT"
# A missing --placer-path/--upsampler-path falls back to the heuristic / Lanczos.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY -m generation.gen_paragraph \
  --device "$DEVICE" \
  --dataset "$DATASET" \
  --save-path "$SAVE_PATH" \
  --style-path "$STYLE_PATH" \
  --stable-dif-path "$STABLE_DIF_PATH" \
  --style-bank-path "$STYLE_BANK" \
  -w "$WRITER_ID" \
  -i "$PROMPT_FILE" \
  -o "$OUTPUT" \
  --placement "${PLACEMENT:-heuristic}" \
  --placer-path "$PLACER_CKPT" \
  --max-line-width "${MAX_LINE_WIDTH:-900}" \
  "$@"
