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

PYTHONPATH="${UNSUPERVISED_AGING}/src/kpms_utils" \
python "${UNSUPERVISED_AGING}/src/kpms_training/create_kpms_project.py" \
    --project_name  "{{project_name}}" \
    --kpms_dir      "${UNSUPERVISED_AGING}/data/kpms_projects/" \
    --videos_dir    "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}/kpms_training_set/videos" \
    --poses_csv_dir "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}/kpms_training_set/poses_csv"

echo "End time: $(date)"
