#!/bin/bash
#SBATCH --job-name=dp-generate-word
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
# Submit from the repo root:  sbatch scripts/slurm/generate_word.sbatch [extra flags]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs

# --- adjust for your cluster (uncomment/edit) ---
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffpen

exec scripts/local/generate_word.sh "$@"
