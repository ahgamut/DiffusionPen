#!/usr/bin/env bash
# Pre-train the style encoder (style_encoder_train.py) -> $STYLE_ENC_SAVE.
# Note: this script uses its own arg style (--batch-size); trains over the
# merged split (MergedWordDataset style_mode). --dataset is only a checkpoint tag.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY style_encoder_train.py \
  --model "${STYLE_MODEL:-mobilenetv2_100}" \
  --dataset "$DATASET" \
  --data-dir "$DATA_DIR" \
  --batch-size "$BATCH_SIZE" \
  --epochs "${STYLE_EPOCHS:-20}" \
  --device "$DEVICE" \
  --style-path "$STYLE_ENC_SAVE" \
  --save-path "$STYLE_ENC_SAVE" \
  "$@"
