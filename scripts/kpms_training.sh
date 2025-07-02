#!/bin/bash
#
#SBATCH --job-name=model_training
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

PROJECT_DIR="/projects/kumar-lab/miaod/projects/unsupervised-aging"
MAMBA_PATH="/projects/compsci/jgeorge/USERS/chouda/miniforge3/envs/keypoint_moseq_gpu"

mamba activate "${MAMBA_PATH}"
python "${PROJECT_DIR}/src/kpms_training.py" \
    --project_name  "2025-07-02_kpms-test" \
    --model_name    "kpms_test" \
    --kpms_dir      "${PROJECT_DIR}/data/kpms_projects/" \
    --videos_dir    "${PROJECT_DIR}/data/datasets/aging_nature-aging/videos" \
    --poses_csv_dir "${PROJECT_DIR}/data/datasets/aging_nature-aging/poses_csv"
