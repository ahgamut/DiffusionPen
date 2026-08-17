#!/usr/bin/env bash
# Build the PLACER memmap split (needs `pip install msgpack`).
# Word-level training data is built by build_multidataset.sh instead.
# Env: set REBUILD=1 to parse raw IAM xml/forms instead of converting placer_IAM.pt.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY -m utils.build_dataset \
  ${REBUILD:+--rebuild} \
  "$@"
