#!/bin/bash
# -------------------------------------------------------------------
# Generate trajectory plots and grid movies from a trained KPMS model.
#
# Produces a syllable similarity dendrogram, per-syllable trajectory
# plots, and grid movies for visualization.
#
# Python: src/feature_extraction/kpms_generate_trajectories.py
#
# Placeholders:
#     {{project_name}}   - Name of the KPMS project
#     {{model_basename}} - Model name prefix (task ID is appended)
#     {{dataset}}        - Name of the target dataset directory
# -------------------------------------------------------------------
#
#SBATCH --job-name=kpms_generate_trajectories
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"

PYTHONPATH="${UNSUPERVISED_AGING}/src/kpms_utils" \
python "${UNSUPERVISED_AGING}/src/feature_extraction/kpms_generate_trajectories.py" \
    --project_name  "{{project_name}}" \
    --model_name    "{{model_basename}}-${SLURM_ARRAY_TASK_ID}" \
    --kpms_dir      "${UNSUPERVISED_AGING}/data/kpms_projects/" \
    --poses_csv_dir "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}/poses_csv"

echo "End time: $(date)"
