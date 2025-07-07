#!/bin/bash
#
#SBATCH --job-name=model_training
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=long
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"

PROJECT_DIR="/projects/kumar-lab/miaod/projects/unsupervised-aging"

source "${PROJECT_DIR}/.venv/bin/activate"
python "${PROJECT_DIR}/src/model_training.py" \
    --input_csv   "${PROJECT_DIR}/data/2025-06-28_kpms-v1_supervised-unsupervised-features.csv" \
    --xcat_json   "${PROJECT_DIR}/data/2025-06-28_kpms-v1_supervised-unsupervised-xcats.json" \
    --output_path "${PROJECT_DIR}/data/2025-06-28_kpms-v1_supervised-unsupervised-results.csv" \
    --seed 623 \
    --outer_n_splits 10 \
    --inner_n_splits 5 \
    --cpu_cores $SLURM_CPUS_ON_NODE \
    --X_cats unsup sup all \
    --y_cats age fi

echo "End time: $(date)"
