#!/bin/bash
#
#SBATCH --job-name=create_kpms_project
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"

PROJECT_DIR="/projects/kumar-lab/miaod/projects/unsupervised-aging"

PYTHONPATH="${PROJECT_DIR}/src/kpms_utils" \
python "${PROJECT_DIR}/src/train_Kpms/create_kpms_project.py" \
    --project_name  "{{project_name}}" \
    --kpms_dir      "${PROJECT_DIR}/data/kpms_projects/" \
    --videos_dir    "${PROJECT_DIR}/data/datasets/{{dataset}}/videos" \
    --poses_csv_dir "${PROJECT_DIR}/data/datasets/{{dataset}}/poses_csv"

echo "End time: $(date)"
