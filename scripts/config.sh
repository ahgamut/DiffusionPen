# Shared configuration for the DiffusionPen run scripts.
#
# Every value is overridable from the environment, e.g.:
#     DEVICE=cuda:1 BATCH_SIZE=64 scripts/local/train_diffusion.sh
# The local/ scripts source this file and cd to the repo root, so they work from
# anywhere. Edit the defaults below to match your box, or export overrides.

# Repo root = parent of this scripts/ dir (resolved from this file's location).
export DP_ROOT="${DP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ---- runtime ----
export DEVICE="${DEVICE:-cuda:0}"          # single-GPU device string
export DATASET="${DATASET:-iam}"           # iam | gnhk | cvl (where supported)

# ---- paths (match the argparse defaults; edit for your box) ----
export SAVE_PATH="${SAVE_PATH:-./diffusionpen_iam_model_path}"
export STYLE_PATH="${STYLE_PATH:-./style_models/iam_style_diffusionpen.pth}"
export STABLE_DIF_PATH="${STABLE_DIF_PATH:-./stable-diffusion-v1-5}"
export STYLE_ENC_SAVE="${STYLE_ENC_SAVE:-./style_models}"

# ---- training hyperparameters ----
export EPOCHS="${EPOCHS:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-320}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LR="${LR:-1e-3}"

# ---- artifacts produced by the aux builders/trainers ----
export STYLE_BANK="${STYLE_BANK:-./saved_iam_data/style_bank.pt}"
export PLACER_CKPT="${PLACER_CKPT:-${SAVE_PATH}/models/placer_seq_ckpt.pt}"
export UPSAMPLER_CKPT="${UPSAMPLER_CKPT:-${SAVE_PATH}/models/upsampler_ckpt.pt}"

# ---- generation ----
export WRITER_ID="${WRITER_ID:-12}"
export PROMPT_FILE="${PROMPT_FILE:-./prompts/sample.txt}"
export SAMPLING_WORD="${SAMPLING_WORD:-hello}"
export OUTPUT="${OUTPUT:-./output.png}"

# ---- python launcher (e.g. "python", "python3", "srun python") ----
export PY="${PY:-python}"
