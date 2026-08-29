#!/bin/bash
#SBATCH --job-name=dp-build-font-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
# CPU-only (PIL rendering, no GPU); font builds are render-bound so give it cores
# and time. Submit from the repo root:
#   sbatch scripts/slurm/build_font_data.sbatch [extra flags]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs

# --- adjust for your cluster (uncomment/edit) ---
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffpen

exec scripts/local/build_font_data.sh "$@"
