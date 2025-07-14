import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns
    import shap
    import xgboost

    from functools import partial
    from pathlib import Path
    from sklearn.base import BaseEstimator
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from typing import Callable, Optional, Sequence

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
    })
    return (
        BaseEstimator,
        Callable,
        Optional,
        Path,
        Sequence,
        json,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        os,
        partial,
        pd,
        r2_score,
        shap,
        xgboost,
    )


@app.cell
def _(Path, json, os, pd):
    unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])
    features_df_path = unsupervised_aging_dir / "data/feature_matrices/2025-07-10_kpms-v2-supervised_feature-matrix.csv"
    xcats_path       = unsupervised_aging_dir / "data/feature_matrices/2025-07-10_kpms-v2-supervised_xcats.json"
    results_df_path  = unsupervised_aging_dir / "data/model_evaluation_results/2025-07-10_kpms-v2-supervised_results.csv"

    features_df = pd.read_csv(features_df_path)
    results_df = pd.read_csv(results_df_path, low_memory=False)
    with xcats_path.open("r") as f:
        xcats = json.load(f)
    xcats.keys()
    return results_df, xcats


@app.cell
def _(
    BaseEstimator,
    Callable,
    Optional,
    Sequence,
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    pd,
    r2_score,
    shap,
):
    def compute_aggregate_shap(
        df: pd.DataFrame,
        model_name: str,
        estimator_ctor: Optional[Callable[[], BaseEstimator]],
        feature_cols: Sequence[str],
        target_col: str = "y_true",
        *,
        num_folds: Optional[int] = None,
        hyperparameter_cols: Optional[Sequence[str]] = None,
        debug: bool = True,
    ):
        df = df[(df["model"] == model_name) & (df["split"] == "test")]
        _print = lambda x: debug and print(x)
        model_name = model_name.lower()

        if len(df) == 0:
            raise ValueError(f"Filtered dataframe has 0 elements.")
    
        if num_folds is None:
            num_folds = int(df["fold"].max()) + 1
            _print(f"Detected {num_folds} folds")

        if hyperparameter_cols is None:
            model_hyperparameter_cols = {
                    "xgboost": [
                       "model__n_estimators", "model__learning_rate", "model__max_depth",
                       "model__subsample", "model__colsample_bytree", "model__min_child_weight",
                       "model__gamma", "model__reg_alpha", "model__reg_lambda",
                   ]
            }
            if model_name not in model_hyperparameter_cols:
                raise ValueError(f"Model name {model_name} not recognized for hyperparameter cols")
            hyperparameter_cols = model_hyperparameter_cols[model_name]
            _print(f"Using model hyperparameters for {model_name}: {hyperparameter_cols}")
        
        integer_hyperparams = ["n_estimators", "max_depth", "min_child_weight"]

        all_shap_dfs = []
        for fold in mo.status.progress_bar(range(num_folds)):
            train_mask = df["fold"] != fold
            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, target_col]
        
            test_mask  = ~train_mask
            X_test  = df.loc[test_mask,  feature_cols]
            y_test  = df.loc[test_mask,  target_col]

            hp_rows = df.loc[test_mask, hyperparameter_cols].drop_duplicates()
        
            if len(hp_rows) != 1:
                raise ValueError(f"Expected a single hyper-parameter row for the test fold but found {len(hp_rows)}.")

            params = {k.replace("model__", ""): v for k, v in hp_rows.iloc[0].items()}
            for k in integer_hyperparams & params.keys():
                params[k] = int(params[k])
            model = estimator_ctor(**params)
            model.fit(X_train, y_train)
        
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            _print(f"Fold {fold}: {{rmse: {rmse:.4f}, mae: {mae:.4f}, R²: {r2:.4f}}}")
        
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer(X_test)
        
            shap_df = pd.DataFrame(shap_values.values, columns=feature_cols, index=X_test.index).assign(fold=fold)
            all_shap_dfs.append(shap_df)

        shap_all = pd.concat(all_shap_dfs, axis=0)
        global_importance = shap_all[feature_cols].abs().mean().sort_values(ascending=False)
        return global_importance, shap_all
    return (compute_aggregate_shap,)


@app.cell
def _(compute_aggregate_shap, partial, results_df, xcats, xgboost):
    _seed = 623

    _estimator_ctor = partial(
        xgboost.XGBRegressor,
        random_state=_seed,
        eval_metric="rmse",
    )

    global_imp, shap_table = compute_aggregate_shap(
        results_df.query("X_cat == 'kpms-v2_all' and y_cat == 'age'"),
        model_name="XGBoost",
        estimator_ctor=_estimator_ctor,
        feature_cols=xcats["kpms-v2_all"]
    )
    return global_imp, shap_table


@app.cell
def _(global_imp, results_df, shap, shap_table):
    _feature_cols = global_imp.index.tolist()
    _X_test_all   = results_df.loc[shap_table.index, _feature_cols]

    shap.summary_plot(
        shap_table[_feature_cols].values,
        _X_test_all,
        feature_names=_feature_cols,
        show=True,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
