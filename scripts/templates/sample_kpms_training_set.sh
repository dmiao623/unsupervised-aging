#!/bin/bash
# -------------------------------------------------------------------
# Sample a stratified subset of videos and pose CSVs for KPMS training.
#
# Draws n_samples records from the dataset's metadata.csv while
# preserving the joint distribution of the specified categorical and
# binned-numeric variables. Not a SLURM job -- run locally or in an
# interactive session.
#
# Python: src/preprocessing/sample_kpms_training_set.py
#
# Placeholders:
#     {{dataset}} - Name of the target dataset directory
# -------------------------------------------------------------------

echo "Start time: $(date)"

python "${UNSUPERVISED_AGING}/src/preprocessing/sample_kpms_training_set.py" \
    --dataset_dir      "${UNSUPERVISED_AGING}/data/datasets/{{dataset}}" \
    --categorical_vars sex diet \
    --strat_vars_bins  age=5 \
    --n_samples        250 \
    --seed             623

echo "End time: $(date)"
