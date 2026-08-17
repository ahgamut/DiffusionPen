#!/usr/bin/env bash
# Train the main latent-diffusion model (train.py).
# Append extra flags, e.g.:  scripts/local/train_diffusion.sh --load-check --no-style-cache
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY train.py \
  --merged-setname "$MERGED_SETNAME" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --save-path "$SAVE_PATH" \
  --style-path "$STYLE_PATH" \
  --stable-dif-path "$STABLE_DIF_PATH" \
  "$@"
