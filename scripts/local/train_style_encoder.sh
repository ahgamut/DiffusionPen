#!/usr/bin/env bash
# Pre-train the style encoder (style_encoder_train.py) -> $STYLE_ENC_SAVE.
# Note: this script uses its own arg style (--batch_size, --data-path, --mode).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY style_encoder_train.py \
  --model "${STYLE_MODEL:-mobilenetv2_100}" \
  --dataset "$DATASET" \
  --data-path "${DATA_PATH:-./iam_data}" \
  --batch_size "$BATCH_SIZE" \
  --epochs "${STYLE_EPOCHS:-20}" \
  --device "$DEVICE" \
  --style-path "$STYLE_ENC_SAVE" \
  --save-path "$STYLE_ENC_SAVE" \
  --mode "${STYLE_MODE:-mixed}" \
  "$@"
