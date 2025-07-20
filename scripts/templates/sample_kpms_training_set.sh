#!/bin/bash

echo "Start time: $(date)"

python "${UNSUPERVISED_AGING}/src/preprocessing/sample_kpms_training_set.py" \
    --dataset_dir "${UNSUPERVISED_AGING}/data/datasets/{{dataset name}}" \
    --categorical_vars sex diet \
    --strat_vars_bins age=5 \
    --n_samples 250 \
    --seed 623

echo "End time: $(date)"
