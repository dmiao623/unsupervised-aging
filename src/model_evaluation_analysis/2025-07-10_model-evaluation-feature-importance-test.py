import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import xgboost as xgb
    import shap
    import os
    import pandas as pd
    from pathlib import Path
    import json

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
    })
    return Path, json, mean_squared_error, np, os, pd, r2_score, shap, xgb


@app.cell
def _(Path, json, os, pd):
    features_df_path = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices/2025-07-10_kpms-v2-supervised_feature-matrix.csv"
    xcats_path       = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices/2025-07-10_kpms-v2-supervised_xcats.json"
    results_df_path  = Path(os.environ["UNSUPERVISED_AGING"]) / "data/model_evaluation_results/2025-07-10_kpms-v2-supervised_results.csv"

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
        (results_df["X_cat"] == "kpms-v2_all") & 
        (results_df["y_cat"] == "fi")
    ]
    return (filt_results_df,)


@app.cell
def _(filt_results_df, mean_squared_error, np, r2_score, shap, xcats, xgb):
    train_mask = filt_results_df["fold"] != 0
    test_mask = filt_results_df["fold"] == 0

    target_col = "y_true"
    X_train = filt_results_df.loc[train_mask, xcats["kpms-v2_all"]]
    y_train = filt_results_df.loc[train_mask, target_col]

    X_test  = filt_results_df.loc[test_mask,  xcats["kpms-v2_all"]]
    y_test  = filt_results_df.loc[test_mask,  target_col]

    xgb_hyperparameters = [
        "model__n_estimators", "model__learning_rate", "model__max_depth",
        "model__subsample",   "model__colsample_bytree", "model__min_child_weight",
        "model__gamma", "model__reg_alpha", "model__reg_lambda",
    ]

    hp_rows = (
        filt_results_df
        .loc[test_mask, xgb_hyperparameters]
        .drop_duplicates()
    )

    if len(hp_rows) != 1:
        raise ValueError(
            f"Expected a single hyper-parameter row for the test fold, "
            f"but found {len(hp_rows)}."
        )

    best_row = hp_rows.iloc[0]

    params = {
        k.replace("model__", ""): v for k, v in best_row.items()
    }
    for k in ("n_estimators", "max_depth", "min_child_weight"):
        params[k] = int(params[k])

    params.update(
        dict(
            objective   = "reg:squarederror",
            booster     = "gbtree",
            random_state= 42,
            n_jobs      = -1,
        )
    )

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"Test RMSE = {rmse:.4f}")
    print(f"Test  R²  = {r2:.4f}")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar")

    shap.summary_plot(shap_values, X_test)
    return X_test, shap_values


@app.cell
def _(shap, shap_values):
    shap.plots.waterfall(shap_values[1])
    return


@app.cell
def _(X_test, shap, shap_values):
    tm_cols = X_test.filter(like="transition_matrix_").columns
    tm_indices = [X_test.columns.get_loc(col) for col in tm_cols]
    shap.summary_plot(shap_values.values[:, tm_indices], X_test[tm_cols], plot_type="bar")
    shap.summary_plot(shap_values.values[:, tm_indices], X_test[tm_cols])

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
