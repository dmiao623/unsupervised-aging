#!/bin/bash
# -------------------------------------------------------------------
# Compare trained KPMS models via expected marginal likelihood.
#
# Computes EML scores for a set of trained models and writes results
# to a JSON file inside the project directory.
#
# Python: src/kpms_training/kpms_model_comparison.py
#
# Placeholders:
#     {{project_name}}   - Name of the KPMS project
#     {{model_basename}} - Model name prefix (1..N is appended)
#     {{num_models}}     - Number of trained models to compare
# -------------------------------------------------------------------
#
#SBATCH --job-name=kpms_model_comparison
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"

python "${UNSUPERVISED_AGING}/src/kpms_training/kpms_model_comparison.py" \
    --project_name   "{{project_name}}" \
    --model_basename "{{model_basename}}" \
    --kpms_dir       "${UNSUPERVISED_AGING}/data/kpms_projects/" \
    --num_models     {{num_models}} \
    --result_path    "${UNSUPERVISED_AGING}/data/kpms_projects/{{project_name}}/model_comparison.json"

echo "End time: $(date)"
