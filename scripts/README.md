# Run scripts

Shell wrappers for the DiffusionPen entry points. Each task has a **local** runner
(`scripts/local/*.sh`, run directly) and a matching **SLURM** batch file
(`scripts/slurm/*.sbatch`, `sbatch`-submitted). The SLURM files just set `#SBATCH`
resources and then `exec` the local script, so the actual command lives in one place.

## Layout

| Task | Local | SLURM | GPU |
|------|-------|-------|-----|
| Build memmap datasets (stage 4) | `local/build_data.sh` | `slurm/build_data.sbatch` | no |
| Build font-augmented split | `local/build_font_data.sh` | `slurm/build_font_data.sbatch` | no |
| Build style bank (stage 3) | `local/build_style_bank.sh` | `slurm/build_style_bank.sbatch` | yes |
| Pre-train style encoder | `local/train_style_encoder.sh` | `slurm/train_style_encoder.sbatch` | yes |
| Train diffusion model | `local/train_diffusion.sh` | `slurm/train_diffusion.sbatch` | yes |
| Train WordPlacer (stage 2) | `local/train_placer.sh` | `slurm/train_placer.sbatch` | yes |
| Train WordUpsampler | `local/train_upsampler.sh` | `slurm/train_upsampler.sbatch` | yes |
| Generate a word | `local/generate_word.sh` | `slurm/generate_word.sbatch` | yes |
| Generate a paragraph | `local/generate_paragraph.sh` | `slurm/generate_paragraph.sbatch` | yes |

## Configuration

All paths/hyperparameters live in `scripts/config.sh` and are **overridable from the
environment** — edit that file for your box, or export overrides per run. The local
scripts `cd` to the repo root, so they work from anywhere. Any extra flags are passed
straight through to the Python script (`"$@"`).

```bash
# edit defaults once:
$EDITOR scripts/config.sh          # SAVE_PATH, STYLE_PATH, STABLE_DIF_PATH, DEVICE, ...

# or override per run:
DEVICE=cuda:1 BATCH_SIZE=64 scripts/local/train_diffusion.sh
scripts/local/train_diffusion.sh --load-check           # extra flag pass-through
```

## Typical order on a fresh box

```bash
pip install msgpack                                      # stage-4 format dep

MULTIDATA_INPUT=./sample-fmt scripts/local/build_multidataset.sh   # -> saved_iam_data/combined_word_train/ (the training data)
# optional: fold in synthetic font writers for glyph/OOV coverage (needs `pip install wordfreq`):
# FONT_DATASETS=csafe,font scripts/local/build_font_data.sh        # real data + one writer per sample-fmt/fonts/*.ttf
# then RE-TRAIN the style encoder on the new split + rebuild its cache/bank before train_diffusion.sh
scripts/local/build_data.sh                              # -> saved_iam_data/iam_placer/ (placer only)
scripts/local/build_style_bank.sh                        # -> saved_iam_data/style_bank.pt  (rebuild to W writers)
scripts/local/train_diffusion.sh                         # -> $SAVE_PATH/models/{ckpt,ema_ckpt}.pt
scripts/local/train_placer.sh                            # -> $PLACER_CKPT
scripts/local/train_upsampler.sh                         # -> $UPSAMPLER_CKPT

# generate (learned layout + upscaling shown; omit for heuristic/Lanczos):
PLACEMENT=learned scripts/local/generate_paragraph.sh --upsample --upsampler-path "$UPSAMPLER_CKPT"
```

On SLURM, submit **from the repo root** so `$SLURM_SUBMIT_DIR` resolves correctly, and
uncomment the `module load` / `conda activate` lines in the `.sbatch` files:

```bash
sbatch scripts/slurm/train_diffusion.sbatch
sbatch scripts/slurm/train_diffusion.sbatch --load-check
```

Logs land in `slurm-logs/<job>-<id>.out`. Adjust `--time`, `--mem`, `--gres` per cluster.

## Notes

- The style bank and learned placement/upsampling **fall back gracefully** when their
  artifact/checkpoint is absent (CNN style path / heuristic layout / Lanczos), so the
  generate scripts work before the aux models are trained.
- `train_style_encoder.sh` uses that script's own flag names (`--batch_size`,
  `--data-path`, `--mode`) — distinct from the others' shared `add_common_args`.
- Multi-GPU (`--dataparallel`) is out of scope for the current single-GPU focus.
