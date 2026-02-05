#!/bin/bash
# -------------------------------------------------------------------
# Run nested cross-validation for regression models.
#
# Evaluates Elastic Net, Random Forest, XGBoost, and MLP regressors
# via nested group-aware cross-validation on the provided feature
# matrix and target variables.
#
# Python: src/model_evaluation/model_evaluation.py
#
# Placeholders:
#     {{feature_matrix}} - Basename of the input feature CSV (no ext)
#     {{xcats}}          - Basename of the X-category JSON (no ext)
#     {{results}}        - Basename of the output CSV (no ext)
#     {{xcat1}}, ...     - Feature group names (keys in the JSON)
# -------------------------------------------------------------------
#
#SBATCH --job-name=model_evaluation
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=long
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --output=/projects/kumar-lab/miaod/projects/unsupervised-aging/logs/output-%j.txt

echo "Start time: $(date)"

python "${UNSUPERVISED_AGING}/src/model_evaluation/model_evaluation.py" \
    --input_csv      "${UNSUPERVISED_AGING}/data/{{feature_matrix}}.csv" \
    --xcat_json      "${UNSUPERVISED_AGING}/data/{{xcats}}.json" \
    --output_path    "${UNSUPERVISED_AGING}/data/{{results}}.csv" \
    --outer_n_splits 10 \
    --inner_n_splits 5 \
    --cpu_cores      $SLURM_CPUS_ON_NODE \
    --X_cats         "{{xcat1}}" "{{xcat2}}" "{{...}}" \
    --y_cats         "age" "fi" \
    --export_all

echo "End time: $(date)"
