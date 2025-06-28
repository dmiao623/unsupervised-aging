#!/bin/bash
#
#SBATCH --job-name=model_training
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=long
#SBATCH --partition=compute
#SBATCH --mem=128G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/data/logs/output-%j.txt

PROJECT_DIR="/projects/kumar-lab/miaod/projects/unsupervised-aging"

source "${PROJECT_DIR}/.venv/bin/activate"
python "${PROJECT_DIR}/src/model-fitting.py" \
    --input_csv   "${PROJECT_DIR}/data/" \
    --xcat_json   "${PROJECT_DIR}/data/" \
    --output_path "${PROJECT_DIR}/results/" \
    --seed 623 \
    --outer_n_splits 50 \
    --inner_n_splits 10 \
    --cpu_cores $SLURM_CPUS_ON_NODE \
    --X_cats body_feats brain_feats \
    --y_cats age fi
