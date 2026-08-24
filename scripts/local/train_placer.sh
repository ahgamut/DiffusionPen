#!/usr/bin/env bash
# Train the stage-2 WordPlacer (placer_seq_train.py) -> $PLACER_CKPT.
# Requires the placer sequence cache (built from placer_IAM.pt / the memmap dir).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY placer_seq_train.py \
  --dataset iam \
  --data-dir "$DATA_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "${PLACER_BATCH:-32}" \
  --lr "$LR" \
  --save-path "$SAVE_PATH" \
  --style-path "$STYLE_PATH" \
  --stable-dif-path "$STABLE_DIF_PATH" \
  "$@"
