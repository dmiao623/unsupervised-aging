# Unsupervised Aging

### Workflow

| Step                              | Description                                                  | Output                                                       |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Keypoint-MoSeq model training     | Sample $n=250$ videos from a dataset and train Keypoint-MoSeq model. Optionally, train multiple models or perform syllable merging | Trained Keypoint-MoSeq model in `data/kpms_projects/<project_name>`. |
| Inference on Full Dataset         | Run inference with trained Keypoint-MoSeq model on entire dataset ([including training data](https://github.com/dattalab/keypoint-moseq/issues/176#issuecomment-2420060339)). | A Keypoint-MoSeq formatted HDF5 in`data/kpms_projects/<project_name>/result.h5`. |
| Feature Extraction                | For each video, extract relevant features (e.g. syllable frequencies, transition probabilities, etc.) into a videos by features Pandas DataFrame. | A `.csv` file storing the feature values with associated metadata. |
| Regression Model Training         | Test a suite of regression models with $K$-fold validation.  | A `.csv` file showing the predicted value of each video.     |
| Analyze Results of Model Training | Aggregate groups to compute $R^2$, MAE, and RMSE and plot these values against one another. |                                                              |

### Sample Directory Structure

```
.
├── data
│   ├── datasets
│   │   ├── dataset1                     # example of a dataset; poses/ is optional
│   │   │   ├── metadata.csv             # metadata file; should contain a "name" identifier column
│   │   │   ├── poses                    # contains *.h5 pose files
│   │   │   ├── poses_csv                # contains *.csv pose files
│   │   │   └── videos                   # contains *.mp4 video files
│   │   └── ...
│   ├── kpms_projects
│   │   ├── kpms_project1                # minimal example of a KPMS project
│   │   │   ├── config.yml
│   │   │   ├── pca.p                    # shared PCA used by all inference instances
│   │   │   ├── result.h5                # combined result used in downstream analysis
│   │   │   ├── kpms_project1_model1
│   │   │   │   └── checkpoint.h5
│   │   │   └── ...                      # each KPMS project may contain several models
│   │   └── ...
│   └── ...                              # other data files (e.g. feature matrices)
├── scripts                              # contains *.sh SLURM scripts
│   └── templates/                       # contains templates for creating SLURM scripts
└── src                                  # contains Python and marimo notebook source files
    ├── kpms_inference/
    ├── kpms_training/
    ├── model_evaluation/
    ├── preprocessing/
    └── kpms_utils -> anshu957/kpms_kumarlab
```

### File Catalog

- Preprocessing
  - `transfer_dataset.py`: creates sym-links of pose and video files to the local `data/datasets` directory.
  - `convert_h5_to_csv.py`: converts HDF5 pose files into CSV poses, stored in a `pose_csv/` directory.
  - `sample_kpms_training_set.py`: samples a subset of videos to be used to train the KPMS model. These videos should be placed in `dataset/kpms_training_set/`. This should be done interactively with marimo.
- KPMS Training
  - `create_kpms_project.py`: creates a KPMS project, including the directory set-up and computing an initial PCA. This should be run on a device with a GPU.
  - `kpms_training.py` trains a KPMS model on the specified dataset. Note that multiple models can be trained simultaneously, with different starting seeds. This should be run on a device with a GPU.
  - `kpms_model_comparison.py`: compares several trained models according to their [Expected Marginal Likelihood scores]([](https://keypoint-moseq.readthedocs.io/en/latest/fitting.html#keypoint_moseq.fitting.expected_marginal_likelihoods)), saving the results as a JSON file. This should be run on a device with a GPU and sufficient memory.
- KPMS Inference
  - `batch_videos.py`: splits the pose files into small batches to be processed in parallel. Note that there are issues when processing $>7$ videos on a single NVIDIA A100 node (see [here](https://github.com/dattalab/keypoint-moseq/issues/176#issuecomment-2421002889)). These should be placed in `dataset/kpms_inference_groups`. Note that the new pose files are sym-linked to the original ones.
  - `kpms_inference.py`: runs the inference algorithm on a single group. The results should be placed in `dataset/kpms_inference_results`. 
  - `merge_batch_results.py`: merges several HDF5 result files into a single HDF5 result file. For downstream analysis, this should be placed in `kpms_project/model_name/result.h5`.
  - `create_index_csv.py`: creates an `index.csv` file that stores group information. This is required by a few KPMS functions to be present.
- Model Evaluation
  - `kpms_syllable_merging`: similar KPMS syllables may be merged to create "metasyllables". This notebooks provides a basic framework for identifying metasyllables and providing manual annotations for them.
  - `model_evaluation.py`: tests several regression models on a specified dataset with a nested cross-validation.
