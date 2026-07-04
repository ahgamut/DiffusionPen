#!/usr/bin/env bash
# Train the stage-2/feature WordUpsampler (upsampler_train.py) -> $UPSAMPLER_CKPT.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY upsampler_train.py \
  --dataset iam \
  --device "$DEVICE" \
  --epochs "${UPS_EPOCHS:-200}" \
  --batch-size "${UPS_BATCH:-64}" \
  --lr "$LR" \
  --scale "${UPS_SCALE:-2}" \
  --num-workers "$NUM_WORKERS" \
  --save-path "$SAVE_PATH" \
  --style-path "$STYLE_PATH" \
  --stable-dif-path "$STABLE_DIF_PATH" \
  "$@"
