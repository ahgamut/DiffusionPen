#!/usr/bin/env bash
# Build the stage-4 memmap dataset dirs (needs `pip install msgpack`).
# Env: BUILD_TARGET=word|placer|cvl|all (default all), SUBSET=train|val|test.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY -m utils.build_dataset \
  --dataset "${BUILD_TARGET:-all}" \
  --subset "${SUBSET:-train}" \
  "$@"
