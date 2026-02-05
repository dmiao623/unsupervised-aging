#!/bin/bash
# -------------------------------------------------------------------
# Train a Keypoint-MoSeq model on preprocessed pose data.
#
# Submits as a SLURM array job; each task trains a separate model
# replicate with the array task ID used as both the model suffix and
# the random seed.
#
# Python: src/kpms_training/kpms_training.py
#
# Placeholders:
#     {{num_models}}     - Number of model replicates to train
#     {{project_name}}   - Name of the KPMS project
#     {{model_basename}} - Model name prefix (task ID is appended)
#     {{dataset}}        - Name of the target dataset directory
# -------------------------------------------------------------------
#
#SBATCH --job-name=kpms_training
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt
#SBATCH --array=1-{{num_models}}

echo "Start time: $(date)"
echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"

PYTHONPATH="${UNSUPERVISED_AGING}/src/kpms_utils" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
TF_FORCE_GPU_ALLOW_GROWTH=true \
python "${UNSUPERVISED_AGING}/src/kpms_training/kpms_training.py" \
    --project_name  "{{project_name}}" \
    --model_name    "{{model_basename}}-${SLURM_ARRAY_TASK_ID}" \
    --kpms_dir      "${UNSUPERVISED_AGING}/data/kpms_projects/" \
    --videos_dir    "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}/kpms_training_set/videos" \
    --poses_csv_dir "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}/kpms_training_set/poses_csv" \
    --seed          "${SLURM_ARRAY_TASK_ID}"

echo "End time: $(date)"
