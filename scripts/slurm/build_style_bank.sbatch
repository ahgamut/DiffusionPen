#!/bin/bash
#SBATCH --job-name=dp-build-style-bank
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
# Submit from the repo root:  sbatch scripts/slurm/build_style_bank.sbatch [extra flags]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs

# --- adjust for your cluster (uncomment/edit) ---
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffpen

exec scripts/local/build_style_bank.sh "$@"
