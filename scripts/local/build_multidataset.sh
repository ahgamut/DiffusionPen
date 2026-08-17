#!/usr/bin/env bash
# Build the merged IAM/CVL/CSAFE word-level memmap split -> the single training
# dataset (MergedWordDataset). Needs `pip install msgpack`.
# Env: MULTIDATA_INPUT (raw folder), MERGED_SETNAME, SPLIT_NAME.
#   e.g. MULTIDATA_INPUT=./my-data SPLIT_NAME=test scripts/local/build_multidataset.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY -m utils.build_multidataset \
  --input "$MULTIDATA_INPUT" \
  --split-name "$SPLIT_NAME" \
  --out-name "${MERGED_SETNAME}_word" \
  "$@"
