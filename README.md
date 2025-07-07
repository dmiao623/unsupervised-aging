# Unsupervised Aging

| Step                              | Description                                                  | Relevant Files                                      | Output                                                       |
| --------------------------------- | ------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ |
| Keypoint-MoSeq model training     | Sample $n=250$ videos from a dataset and train Keypoint-MoSeq model. Optionally, train multiple models or perform syllable merging | `src/create_kpms_project.py`, `src/train_kpms.py`   | Trained Keypoint-MoSeq model                                 |
| Inference on Full Dataset         | Run inference with trained Keypoint-MoSeq model on entire dataset ([including training data](https://github.com/dattalab/keypoint-moseq/issues/176#issuecomment-2420060339)). |                                                     | One or more `.h5` result files                               |
| Feature Extraction                | For each video, extract relevant features (e.g. syllable frequencies, transition probabilities, etc.) into a videos by features Pandas DataFrame. |                                                     | A `.csv` file storing the feature values with associated metadata. |
| Regression Model Training         | Test a suite of regression models with $K$-fold validation.  | `src/model_training.py`                             | A `.csv` file showing the predicted value of each video.     |
| Analyze Results of Model Training | Aggregate groups to compute $R^2$, MAE, and RMSE and plot these values against one another. | `src/2025-07-01_model-training-data-analysis.ipynb` | Plots (saved in `notebooks/`).                               |

