#!/bin/bash
#
#SBATCH --job-name=train_multi_kpms
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt
#SBATCH --array=1-20

echo "Start time: $(date)"
echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"

PROJECT_DIR="/projects/kumar-lab/miaod/projects/unsupervised-aging"

PYTHONPATH="${PROJECT_DIR}/src/kpms_kumarlab" \
python "${PROJECT_DIR}/src/train_kpms.py" \
    --project_name  "2025-07-03_kpms-v2" \
    --model_name    "2025-07-03_model-${SLURM_ARRAY_TASK_ID}" \
    --kpms_dir      "${PROJECT_DIR}/data/kpms_projects/" \
    --videos_dir    "${PROJECT_DIR}/data/datasets/aging_nature-aging/videos" \
    --poses_csv_dir "${PROJECT_DIR}/data/datasets/aging_nature-aging/poses_csv" \
    --seed          "${SLURM_ARRAY_TASK_ID}"

echo "End time: $(date)"
