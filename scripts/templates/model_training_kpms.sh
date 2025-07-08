#!/bin/bash
#
#SBATCH --job-name=model_training
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=long
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"

python "${UNSUPERVISED_AGING}/src/model_fitting/model_fitting.py" \
    --input_csv   "${UNSUPERVISED_AGING}/data/{{feature matrix}}.csv" \
    --xcat_json   "${UNSUPERVISED_AGING}/data/{{xcats}}.json" \
    --output_path "${UNSUPERVISED_AGING}/data/{{results}}.csv" \
    --outer_n_splits 10 \
    --inner_n_splits 5 \
    --cpu_cores $SLURM_CPUS_ON_NODE \
    --X_cats "{{xcat1}}" "{{xcat2}}" "{{...}}"  \
    --y_cats "age" "fi"

echo "End time: $(date)"
