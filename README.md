# Unsupervised Aging

An unsupervised behavioral phenotyping pipeline for studying aging in mice using [Keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/) (KPMS). The pipeline processes pose-estimation data from video recordings of Diversity Outbred (DO) and C57BL/6 (B6) mice, learns behavioral syllables via an autoregressive hidden Markov model, and extracts feature matrices for downstream regression analysis of aging and frailty.

## Pipeline Overview

```
Pose Estimation (H5) ──► Preprocessing ──► KPMS Training ──► Inference
                          (sampling,        (AR-HMM,         (apply model
                           conversion)       full model)       to new data)
                                                │
                                                ▼
                         Model Evaluation ◄── Feature Extraction
                         (nested CV with       (syllable frequencies,
                          grouped splits)       latent embeddings)
```

## Project Structure

```
unsupervised-aging/
├── src/
│   ├── preprocessing/          # Data preparation and stratified sampling
│   ├── kpms_training/          # Keypoint-MoSeq model training
│   ├── kpms_inference/         # Batch inference on new datasets
│   ├── feature_extraction/     # Behavioral feature computation
│   └── model_evaluation/       # Nested cross-validation regression
├── scripts/templates/          # SLURM job submission templates
├── figures/                    # Output visualizations
└── pyproject.toml              # Project metadata and dependencies
```

### Preprocessing (`src/preprocessing/`)

| Script | Description |
|--------|-------------|
| `convert_h5_to_csv.py` | Convert pose-estimation H5 files to CSV |
| `sample_kpms_training_set.py` | Stratified sampling preserving joint distribution of categorical/numeric variables |
| `sample_combined_kpms_training_set.py` | Strain-balanced sampling across DO and B6 |
| `sample_young_old_kpms_training_set.py` | Age-filtered sampling with young/old splits based on quantile thresholds |
| `transfer_dataset.py` | Symlink trimmed videos and pose files into a dataset directory |
| `transfer_dataset_from_file.py` | Symlink pose H5 files from a text file listing |

### KPMS Training (`src/kpms_training/`)

| Script | Description |
|--------|-------------|
| `create_kpms_project.py` | Initialize a KPMS project: set bodyparts/skeleton, run PCA |
| `kpms_training.py` | Train a KPMS model (AR-HMM then full model), reindex syllables, export results |
| `kpms_model_comparison.py` | Compute Expected Marginal Likelihood scores across model replicates |
| `kpms_training_utils.py` | Shared training utilities (`fit_and_save_model`) |

### KPMS Inference (`src/kpms_inference/`)

| Script | Description |
|--------|-------------|
| `kpms_inference.py` | Apply a trained checkpoint to new pose data, write HDF5 results |
| `batch_videos.py` | Distribute files into sub-folders for memory-constrained batching |
| `create_index_csv.py` | Create placeholder `index.csv` mapping each pose file to its own group |
| `merge_batch_results.py` | Merge per-batch HDF5 result files into a single `results.h5` |
| `merge_batch_results_no_dups.py` | Merge variant that skips duplicate groups |

### Feature Extraction (`src/feature_extraction/`)

| Script | Description |
|--------|-------------|
| `feature_extraction.py` | Compute latent-embedding statistics, syllable frequencies, and optional meta-syllable transitions; merge with metadata and supervised features |
| `kpms_generate_trajectories.py` | Generate dendrograms, per-syllable trajectory plots, and grid movies |
| `create_dendrogram.py` | Compute and save a syllable similarity dendrogram |
| `kpms_syllable_merging.py` | Hierarchical syllable merging via agglomerative clustering (marimo notebook) |

### Model Evaluation (`src/model_evaluation/`)

| Script | Description |
|--------|-------------|
| `model_evaluation.py` | Nested group-aware cross-validation with Elastic Net, Random Forest, XGBoost, and MLP regressors |
| `model_evaluation_utils.py` | `TwoStageSearchCV` for sequential hyperparameter tuning; `compute_nested_kfold_validation` for nested CV |
| `stack_evaluation_results.py` | Concatenate multiple evaluation CSVs into a single output |

## Requirements

- Python >= 3.10
- [Keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/) and JAX (for training and inference)
- SLURM-managed HPC cluster (for GPU-accelerated model training)

Python dependencies are managed with [uv](https://docs.astral.sh/uv/) and defined in `pyproject.toml`:

```
h5py          >= 3.14.0
numpy         >= 2.2.6
pandas        >= 2.3.2
scikit-learn  >= 1.7.2
xgboost       >= 3.0.5
matplotlib    >= 3.10.6
marimo        >= 0.16.2
tqdm          >= 4.67.1
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd unsupervised-aging
   ```

2. **Create the virtual environment:**
   ```bash
   uv sync
   ```

3. **Prepare data:** Convert pose-estimation H5 files to CSV and draw a stratified training sample:
   ```bash
   python src/preprocessing/convert_h5_to_csv.py --dataset_dir <path>
   python src/preprocessing/sample_kpms_training_set.py \
       --dataset_dir <path> --export_dir <path> --n_samples 200
   ```

4. **Initialize a KPMS project and train:**
   ```bash
   python src/kpms_training/create_kpms_project.py \
       --project_name my_project --kpms_dir <path> \
       --videos_dir <path> --poses_csv_dir <path>

   python src/kpms_training/kpms_training.py \
       --project_name my_project --model_name model_v1 \
       --kpms_dir <path> --videos_dir <path> --poses_csv_dir <path>
   ```

5. **Run inference on new data:**
   ```bash
   python src/kpms_inference/kpms_inference.py \
       --project_name my_project --model_name model_v1 \
       --kpms_dir <path> --poses_csv_dir <path>
   ```

6. **Extract features and evaluate:**
   ```bash
   python src/feature_extraction/feature_extraction.py \
       --project_name my_project --model_name model_v1 \
       --kpms_dir <path> --dataset_dir <path> \
       --adj_metadata_path <metadata.csv> --output_dir <path>

   python src/model_evaluation/model_evaluation.py \
       --input_csv <features.csv> --xcat_json <categories.json> \
       --output_path <results.csv>
   ```

## SLURM Templates

Job submission scripts for each pipeline stage are in `scripts/templates/`. Each template contains placeholder paths and resource configurations for NVIDIA A100 GPUs:

| Template | Resources | Purpose |
|----------|-----------|---------|
| `convert_h5_to_csv.sh` | 3h, 8 CPU, 1 GPU (MIG) | H5 to CSV conversion |
| `create_kpms_project.sh` | 1h, 8 CPU, 1 GPU (A100) | Project initialization and PCA |
| `kpms_training.sh` | 12h, array job, 8 CPU, 1 GPU (A100) | Model training (task ID = seed) |
| `kpms_inference.sh` | 1.5h, array job, 8 CPU, 1 GPU (A100) | Batch inference |
| `kpms_model_comparison.sh` | 4h, 8 CPU, 1 GPU (A100) | EML score computation |
| `model_evaluation.sh` | 1 day, 64 CPU | Nested CV regression |
