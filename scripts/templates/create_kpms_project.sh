#!/bin/bash
# -------------------------------------------------------------------
# Initialize a new Keypoint-MoSeq project directory.
#
# Sets up the project structure, configures bodyparts and skeleton,
# loads pose data, and runs PCA to select latent dimensionality.
#
# Python: src/kpms_training/create_kpms_project.py
#
# Placeholders:
#     {{project_name}} - Name for the new KPMS project
#     {{dataset}}      - Name of the target dataset directory
# -------------------------------------------------------------------
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
