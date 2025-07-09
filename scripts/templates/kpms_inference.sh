#!/bin/bash
#
#SBATCH --job-name=kpms_inference
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_training
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt
#SBATCH --array=1-{{num groups}}

echo "Start time: $(date)"
echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"

PYTHONPATH="${UNSUPERVISED_AGING}/src/kpms_utils" \
python "${UNSUPERVISED_AGING}/src/kpms_inference/kpms_inference.py" \
    --project_name  "{{project name}}" \
    --model_name    "{{model basename}}-${SLURM_ARRAY_TASK_ID}" \
    --kpms_dir      "${UNSUPERVISED_AGING}/data/kpms_projects/" \
    --poses_csv_dir "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}/poses_csv" \
    --result_path   "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}/inference_results/result-${SLURM_ARRAY_TASK_ID}.h5"

echo "End time: $(date)"
