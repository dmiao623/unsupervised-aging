import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import numpy as np
    import os
    import pandas as pd
    import shap

    from pathlib import Path
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBRegressor

    # mpl.style.use("default")
    return Path, XGBRegressor, json, np, pd, shap


@app.cell
def _(Path, json, pd):
    xcats_path       = Path("/projects/kumar-lab/miaod/projects/unsupervised-aging/data/feature_matrices/2025-07-23_xcats__combined_1126__2025-07-20_kpms-v4_150__2025-07-20_model-1.json")
    features_df_path  = Path("/projects/kumar-lab/miaod/projects/unsupervised-aging/data/feature_matrices/2025-07-23_feature-matrix__combined_1126__2025-07-20_kpms-v4_150__2025-07-20_model-1.csv")
    results_df_path = Path("/projects/kumar-lab/miaod/projects/unsupervised-aging/data/model_evaluation_results/2025-07-23_model-evaluation-results__combined_1126__2025-07-20_kpms-v4_150__2025-07-20_model-1.csv")

    features_df = pd.read_csv(features_df_path)
    results_df = pd.read_csv(results_df_path, low_memory=False)
    with xcats_path.open("r") as f:
        xcats = json.load(f)
    xcats.keys()
    return results_df, xcats


@app.cell
def _(results_df):
    filt_results_df = results_df[
        (results_df["model"] == "XGBoost") &
        (results_df["split"] == "test") & 
        (results_df["X_cat"] == "all") &
        (results_df["y_cat"] == "age")
    ]
    list(filt_results_df.columns)
    return (filt_results_df,)


@app.cell
def _(filt_results_df):
    test_mask = filt_results_df["fold"] == 0
    train_mask = filt_results_df["fold"] != 0
    xgb_hyperparameters = ["model__n_estimators", "model__learning_rate", "model__max_depth", "model__subsample", "model__colsample_bytree", "model__min_child_weight", "model__gamma", "model__reg_alpha", "model__reg_lambda"]

    filt_results_df[test_mask][xgb_hyperparameters]
    return test_mask, train_mask, xgb_hyperparameters


@app.cell
def _(
    XGBRegressor,
    filt_results_df,
    np,
    shap,
    test_mask,
    train_mask,
    xcats,
    xgb_hyperparameters,
):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    xcat = xcats["unsupervised"]
    target_col = "y_true"
    X_train = filt_results_df.loc[train_mask, xcat]
    y_train = filt_results_df.loc[train_mask, target_col]

    X_test  = filt_results_df.loc[test_mask,  xcat]
    y_test  = filt_results_df.loc[test_mask,  target_col]

    # ────────────────────────────────────────────────────────────────────────────────
    # 2.  Extract the tuned hyper-parameters that belong to the test fold (fold == 0)
    #     • Strip the "model__" prefix so XGBoost recognises them
    #     • Cast ints where XGBoost expects integers
    # ────────────────────────────────────────────────────────────────────────────────
    # xgb_hyperparameters = [
    #     "model__n_estimators", "model__learning_rate", "model__max_depth",
    #     "model__subsample",   "model__colsample_bytree", "model__min_child_weight",
    #     "model__gamma", "model__reg_alpha", "model__reg_lambda"
    # ]

    hp_rows = (
        filt_results_df
        .loc[test_mask, xgb_hyperparameters]
        .drop_duplicates()
    )

    # (ii) sanity-check: should be exactly one unique combination
    if len(hp_rows) != 1:
        raise ValueError(
            f"Expected a single hyper-parameter row for the test fold, "
            f"but found {len(hp_rows)}."
        )

    # (iii) convert that single row to a Series
    best_row = hp_rows.iloc[0]          # ← one-dimensional Series

    # strip the “model__” prefix so XGBoost recognises them
    params = {
        k.replace("model__", ""): v
        for k, v in best_row.items()
    }

    # cast the parameters XGBoost insists be integers
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    # optionally make the objective explicit and fix randomness
    params.update(
        dict(
            objective   = "reg:squarederror",   # classic squared-error regression
            booster     = "gbtree",
            random_state= 42,
            n_jobs      = -1,
        )
    )

    # ────────────────────────────────────────────────────────────────────────────────
    # 3.  Train and evaluate
    # ────────────────────────────────────────────────────────────────────────────────
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"Test RMSE = {rmse:.4f}")
    print(f"Test  R²  = {r2:.4f}")

    # ────────────────────────────────────────────────────────────────────────────────
    # 4.  SHAP (Shapley) value analysis – identical workflow
    # ────────────────────────────────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # shape: (n_rows, n_features)

    # — Global importance —
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # — Distribution + direction —
    shap.summary_plot(shap_values, X_test)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
