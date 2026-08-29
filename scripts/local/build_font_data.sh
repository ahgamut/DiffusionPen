#!/usr/bin/env bash
# Build a font-augmented merged split: real datasets + synthetic "font writers"
# (one writer per .ttf/.otf in FONT_DIR, words PIL-rendered + augmented for glyph
# coverage -- see utils/font_synth.py). Needs `pip install msgpack wordfreq`.
# Env: FONT_DIR, FONT_DATASETS (e.g. "csafe,font" or just "font"), MULTIDATA_INPUT,
#      MERGED_SETNAME, SPLIT_NAME, FONT_WORDS_PER_WRITER, FONT_INSTANCES_PER_WORD.
#   # font-only smoke build (no real data needed; --input is ignored for font-only):
#   FONT_DATASETS=font SPLIT_NAME=fonttest scripts/local/build_font_data.sh \
#       --font-words-per-writer 50 --font-instances-per-word 2
#   # the real target -- CSAFE + fonts:
#   FONT_DATASETS=csafe,font scripts/local/build_font_data.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
cd "$DP_ROOT"

$PY -m utils.build_multidataset \
  --datasets "$FONT_DATASETS" \
  --input "$MULTIDATA_INPUT" \
  --font-dir "$FONT_DIR" \
  --split-name "$SPLIT_NAME" \
  --out-name "${MERGED_SETNAME}_word" \
  --font-words-per-writer "$FONT_WORDS_PER_WRITER" \
  --font-instances-per-word "$FONT_INSTANCES_PER_WORD" \
  "$@"
