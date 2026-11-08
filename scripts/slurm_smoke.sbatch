#!/bin/bash
#SBATCH --job-name=sear-smoke
#SBATCH --output=sear_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# The partition/account/GPU flags are CLUSTER-SPECIFIC: customize for Fir or Speed.
set -euo pipefail
source /scratch/onejati/venvs/medqwenv2/bin/activate
cd "${SLURM_SUBMIT_DIR}"
python -m pytest -q tests/
python scripts/check_qwen.py --random-init
