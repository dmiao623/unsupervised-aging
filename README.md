# Unsupervised Aging

### Workflow

| Step                              | Description                                                  | Relevant Files                                      | Output                                                       |
| --------------------------------- | ------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ |
| Keypoint-MoSeq model training     | Sample $n=250$ videos from a dataset and train Keypoint-MoSeq model. Optionally, train multiple models or perform syllable merging | `src/create_kpms_project.py`, `src/train_kpms.py`   | Trained Keypoint-MoSeq model                                 |
| Inference on Full Dataset         | Run inference with trained Keypoint-MoSeq model on entire dataset ([including training data](https://github.com/dattalab/keypoint-moseq/issues/176#issuecomment-2420060339)). |                                                     | One or more `.h5` result files                               |
| Feature Extraction                | For each video, extract relevant features (e.g. syllable frequencies, transition probabilities, etc.) into a videos by features Pandas DataFrame. |                                                     | A `.csv` file storing the feature values with associated metadata. |
| Regression Model Training         | Test a suite of regression models with $K$-fold validation.  | `src/model_training.py`                             | A `.csv` file showing the predicted value of each video.     |
| Analyze Results of Model Training | Aggregate groups to compute $R^2$, MAE, and RMSE and plot these values against one another. | `src/2025-07-01_model-training-data-analysis.ipynb` | Plots (saved in `notebooks/`).                               |

### Directory Structure

```
.
├── data
│   ├── datasets
│   │   ├── dataset1                     # example of a dataset; poses/ is optional
│   │   │   ├── poses                    # contains *.h5 pose files
│   │   │   ├── poses_csv                # contains *.csv pose files
│   │   │   └── videos                   # contains *.mp4 video files
│   │   └── ...
│   ├── kpms_projects
│   │   ├── kpms_project1                # minimal example of a KPMS project
│   │   │   ├── config.yml
│   │   │   ├── pca.p
│   │   │   ├── kpms_project1_model1
│   │   │   │   └── checkpoint.h5
│   │   │   └── ...                      # each KPMS project may contain several models
│   │   └── ...
│   └── ...                              # other data files (e.g. feature matrices)
├── scripts/                             # contains *.sh SLURM scripts
└── src                                  # contains Python and marimo notebook source files
    ├── kpms_inference
    │   ├── batch_videos.py              # preprocesses videos into batches to be run in parallel
    │   ├── kpms_inference.py            # runs inference on each batch
    │   └── merge_results.py             # merges seperate *.h5 result files from each batch
    ├── model_fitting
    │   ├── model_training.py            # performs a K-fold validation over several regerssion models
    │   └── model_training_utils.py
    └── train_kpms
    │   ├── create_kpms_project.py       # initializes a KPMS project and runs preprocessing and PCA
    │   ├── train_kpms.py                # trains one or more KPMS model
    │   └── train_kpms_utils.py
    └── kpms_utils -> anshu957/kpms_kumarlab
```

