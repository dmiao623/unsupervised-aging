#!/bin/bash
#
#SBATCH --job-name=convert_h5_to_csv
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_inference
#SBATCH --partition=gpu_a100_mig
#SBATCH --mem=126G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"

PYTHONPATH="${UNSUPERVISED_AGING}/src/kpms_utils" \
python "${UNSUPERVISED_AGING}/src/preprocessing/convert_h5_to_csv.py" \
    --dataset_dir "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}" \
    --strict_mode

echo "End time: $(date)"
