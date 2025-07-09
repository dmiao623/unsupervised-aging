#!/bin/bash
#
#SBATCH --job-name=kpms_generate_trajectories
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"
echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"

PYTHONPATH="${UNSUPERVISED_AGING}/src/kpms_utils" \
python "${UNSUPERVISED_AGING}/src/model_evaluation/kpms_generate_trajectories.py" \
    --project_name  "2025-07-03_kpms-v2" \
    --model_name    "2025-07-07_model-2" \
    --kpms_dir      "${UNSUPERVISED_AGING}/data/kpms_projects/" \
    --videos_dir    "${UNSUPERVISED_AGING}/data/datasets/nature-aging_370/videos" \
    --poses_csv_dir "${UNSUPERVISED_AGING}/data/datasets/nature-aging_370/poses_csv"

echo "End time: $(date)"
