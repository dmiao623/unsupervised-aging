import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import seaborn as sns
    import shap

    from pathlib import Path
    from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Sequence
    from scipy.stats import rankdata
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor

    mpl.style.use("default")
    return (
        Callable,
        Optional,
        Path,
        Tuple,
        XGBRegressor,
        json,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        os,
        pd,
        plt,
        r2_score,
        rankdata,
        shap,
        sns,
    )


@app.cell
def _(Path, json, os, pd):
    unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])
    data_info_path         = unsupervised_aging_dir / "data/data_info.json"

    with data_info_path.open("r") as f:
        data_info = json.load(f)

    data = {}
    for dataset, info in data_info.items():
        with Path(info["xcats_path"]).open("r") as f:
            xcats = json.load(f)
        data[dataset] = {
            "features": pd.read_csv(info["features_path"], index_col=0),
            "xcats": xcats,
            "results": pd.read_csv(info["model_eval_path"], index_col=0, low_memory=False)
        }
    data_info
    return (data,)


@app.cell
def _(Callable, Tuple, mean_absolute_error, mean_squared_error, pd, r2_score):
    Metric_t = Tuple[Callable[[pd.Series, pd.Series], float], str, bool]

    mae_  = (mean_absolute_error, "MAE", True)
    rmse_ = (mean_squared_error, "RMSE", True)
    r2_   = (r2_score, "R²", False)
    return (Metric_t,)


@app.cell
def _(Metric_t, Optional, Tuple, data, np, pd, plt, sns):
    def plot_foldwise_errors(
        y_cat: str,
        metric: Metric_t,
        *,
        model: Optional[str]     = None,
        figsize: Tuple[int, int] = (10, 6),
        x_cats: Tuple[str, ...]  = ("unsupervised", "supervised", "all"),
        palette: str             = "Set1",
        title: str               = "",
        y_label: str             = "Error",
        x_label: str             = "Group",
    ) -> None:
        plot_data = []
        metric_fn, metric_name, is_loss = metric

        for label, dcs in zip(("B6J", "DO", "B6J/DO"), data.values()):
            results_all = dcs["results"].query('split == "test" and y_cat == @y_cat')

            for X_cat in x_cats:
                if model is None:
                    best_model = None
                    best_value = None

                    for m in results_all["model"].unique():
                        results_model = results_all.query('model == @m and X_cat == @X_cat')
                        values = []
                        for fold in results_model["fold"].unique():
                            fold_data = results_model[results_model["fold"] == fold]
                            val = metric_fn(fold_data["y_true"], fold_data["y_pred"])
                            values.append(val)

                        if values:
                            median_val = np.median(values)
                            if best_value is None or (median_val < best_value if is_loss else median_val > best_value):
                                best_value = median_val
                                best_model = m

                    print(f"[DEBUG] Selected model for group='{label}', X_cat='{X_cat}': {best_model}")
                    results = results_all.query('model == @best_model and X_cat == @X_cat')
                else:
                    results = results_all.query('model == @model and X_cat == @X_cat')

                for fold in results["fold"].unique():
                    fold_data = results[results["fold"] == fold]
                    err = metric_fn(fold_data["y_true"], fold_data["y_pred"])
                    plot_data.append({
                        y_label: err,
                        x_label: label,
                        "X_cat": X_cat
                    })

        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=figsize)

        sns.boxplot(data=df_plot, x=x_label, y=y_label, hue="X_cat", palette=palette)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return


@app.cell
def _(Optional, Tuple, XGBRegressor, data, mo, np, pd, plt, shap, sns):
    def plot_shap_mean_abs_prediction(
        dataset_label: str,
        X_cat: str,
        y_cat: str,
        *,
        top_n: Optional[int]                 = None,
        model: str                           = "XGBoost",
        figsize: Tuple[int, int]             = (6, 6),
        title: str                           = "",
        y_label: str                         = "Feature",
        x_label: str                         = "SHAP value",
        x_lim: Optional[Tuple[float, float]] = None
    ) -> None:

        results = data[dataset_label]["results"].query(
            'split == "test" and model == @model and X_cat == @X_cat and y_cat == @y_cat'
        )
        features = data[dataset_label]["xcats"][X_cat]

        xgb_hyperparameters = [
            ("model__n_estimators", int),
            ("model__learning_rate", float),
            ("model__max_depth", int),
            ("model__subsample", float),
            ("model__colsample_bytree", float),
            ("model__min_child_weight", float),
            ("model__gamma", float),
            ("model__reg_alpha", float),
            ("model__reg_lambda", float)
        ]

        num_folds = max(results["fold"]) + 1
        print(f"[DEBUG] Detected {num_folds} validation folds.")

        all_shap_values = []
        for fold in mo.status.progress_bar(range(num_folds)):
            train_mask = results["fold"] != fold
            test_mask  = results["fold"] == fold

            X_train = results.loc[train_mask, features]
            y_train = results.loc[train_mask, "y_true"]

            X_test  = results.loc[test_mask,  features]
            y_test  = results.loc[test_mask,  "y_true"]

            hp_rows = results.loc[test_mask, [name for name, _ in xgb_hyperparameters]].drop_duplicates()
            if len(hp_rows) != 1:
                raise ValueError(f"Expected a single hyper-parameter row for the test fold but found {len(hp_rows)}.")
            params = hp_rows.iloc[0].to_dict()
            for param_name, param_type in xgb_hyperparameters:
                params[param_name] = param_type(params[param_name])
            params = {k.replace("model__", ""): v for k, v in params.items()}

            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            explainer = shap.Explainer(model)
            shap_vals = explainer(X_test)

            all_shap_values.append(np.abs(shap_vals.values))

        shap_matrix = np.vstack(all_shap_values)
        mean_abs_shap = shap_matrix.mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=features).sort_values(ascending=False)

        if top_n is not None:
            feature_importance = feature_importance.head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(
            x=feature_importance.values,
            y=feature_importance.index,
            color="steelblue"
        )
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_lim is not None:
            plt.xlim(x_lim)
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(Optional, Tuple, XGBRegressor, data, mo, np, pd, plt, rankdata, shap):
    def plot_shap_summary_dot(
        dataset_label: str,
        X_cat: str,
        y_cat: str,
        *,
        top_n: Optional[int]                 = None,
        model: str                           = "XGBoost",
        figsize: Tuple[int, int]             = (8, 6),
        title: str                           = "",
        y_label: str                         = "Feature",
        x_label: str                         = "SHAP value",
        x_lim: Optional[Tuple[float, float]] = None,
        cbar_label: str                      = "Feature Value Percentile"
    ) -> None:
        results = data[dataset_label]["results"].query(
            'split == "test" and model == @model and X_cat == @X_cat and y_cat == @y_cat'
        )
        features = data[dataset_label]["xcats"][X_cat]

        xgb_hyperparameters = [
            ("model__n_estimators", int),
            ("model__learning_rate", float),
            ("model__max_depth", int),
            ("model__subsample", float),
            ("model__colsample_bytree", float),
            ("model__min_child_weight", float),
            ("model__gamma", float),
            ("model__reg_alpha", float),
            ("model__reg_lambda", float)
        ]

        num_folds = max(results["fold"]) + 1
        print(f"[DEBUG] Detected {num_folds} validation folds.")
    
        all_shap_values = []
        all_feature_values = []
        for fold in mo.status.progress_bar(range(num_folds)):
            train_mask = results["fold"] != fold
            test_mask  = results["fold"] == fold

            X_train = results.loc[train_mask, features]
            y_train = results.loc[train_mask, "y_true"]

            X_test  = results.loc[test_mask, features]
            y_test  = results.loc[test_mask, "y_true"]

            hp_rows = results.loc[test_mask, [name for name, _ in xgb_hyperparameters]].drop_duplicates()
            if len(hp_rows) != 1:
                raise ValueError(f"Expected a single hyper-parameter row for the test fold but found {len(hp_rows)}.")
            params = hp_rows.iloc[0].to_dict()
            for param_name, param_type in xgb_hyperparameters:
                params[param_name] = param_type(params[param_name])
            params = {k.replace("model__", ""): v for k, v in params.items()}

            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            explainer = shap.Explainer(model)
            shap_vals = explainer(X_test)
            all_shap_values.append(shap_vals.values)
            all_feature_values.append(X_test)

        shap_matrix = np.vstack(all_shap_values)
        feature_matrix = pd.concat(all_feature_values, axis=0)
        feature_names = features

        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        feature_order = np.argsort(mean_abs_shap)[::-1]
        if top_n is not None:
            feature_order = feature_order[:top_n]

        cmap = plt.get_cmap("coolwarm")
        fig, ax = plt.subplots(figsize=figsize)

        y_ticks = []
        y_labels = []
        for i, feature_idx in enumerate(feature_order):
            shap_vals = shap_matrix[:, feature_idx]
            feature_vals = feature_matrix.iloc[:, feature_idx]
            ranks = rankdata(feature_vals)
            norm_vals = (ranks - 1) / (len(ranks) - 1 + 1e-8)
            colors = cmap(norm_vals)

            jitter = np.random.uniform(-0.2, 0.2, size=len(shap_vals))
            y_pos = np.full_like(shap_vals, fill_value=i, dtype=float) + jitter

            ax.scatter(shap_vals, y_pos, color=colors, s=10, alpha=0.7, edgecolors='none')
            y_ticks.append(i)
            y_labels.append(feature_names[feature_idx])

        ax.set_yticks(y_ticks[::-1])
        ax.set_yticklabels(y_labels[::-1])
        ax.invert_yaxis()

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cbar_label)

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(Optional, Tuple, XGBRegressor, data, mo, np, pd, plt, shap, sns):
    from scipy.stats import ttest_ind


    def plot_differential_B6J_DO_shap_mean_abs_predictions(
        dataset_label: str,
        X_cat: str,
        y_cat: str,
        *,
        top_n: Optional[int]                 = None,
        model: str                           = "XGBoost",
        figsize: Tuple[int, int]             = (6, 6),
        title: str                           = "",
        y_label: str                         = "Feature",
        x_label: str                         = "Δ SHAP value",
        x_lim: Optional[Tuple[float, float]] = None
    ) -> None:

        results = data[dataset_label]["results"].query(
            'split == "test" and model == @model and X_cat == @X_cat and y_cat == @y_cat'
        ).copy()

        results["strain"] = data[dataset_label]["features"].loc[results["sample_id"], "strain"].values
    
        features = data[dataset_label]["xcats"][X_cat]

        xgb_hyperparameters = [
            ("model__n_estimators", int),
            ("model__learning_rate", float),
            ("model__max_depth", int),
            ("model__subsample", float),
            ("model__colsample_bytree", float),
            ("model__min_child_weight", float),
            ("model__gamma", float),
            ("model__reg_alpha", float),
            ("model__reg_lambda", float)
        ]

        num_folds = max(results["fold"]) + 1
        print(f"[DEBUG] Detected {num_folds} validation folds.")

        shap_group1 = []
        shap_group2 = []

        for fold in mo.status.progress_bar(range(num_folds)):
            train_mask = results["fold"] != fold
            test_mask  = results["fold"] == fold

            X_train = results.loc[train_mask, features]
            y_train = results.loc[train_mask, "y_true"]

            X_test  = results.loc[test_mask,  features]
            y_test  = results.loc[test_mask,  "y_true"]

            hp_rows = results.loc[test_mask, [name for name, _ in xgb_hyperparameters]].drop_duplicates()
            if len(hp_rows) != 1:
                raise ValueError(f"Expected a single hyper-parameter row for the test fold but found {len(hp_rows)}.")
            params = hp_rows.iloc[0].to_dict()
            for param_name, param_type in xgb_hyperparameters:
                params[param_name] = param_type(params[param_name])
            params = {k.replace("model__", ""): v for k, v in params.items()}

            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            explainer = shap.Explainer(model)
            shap_vals = explainer(X_test)

            df_test = results.loc[test_mask].copy()
            shap_abs = np.abs(shap_vals.values)
            df_shap = pd.DataFrame(shap_abs, columns=features, index=df_test.index)

            mask1 = df_test.query('strain == "B6"').index
            mask2 = df_test.query('strain == "DO"').index

            shap_group1.append(df_shap.loc[mask1].mean().values)
            shap_group2.append(df_shap.loc[mask2].mean().values)

        group1_matrix = np.vstack(shap_group1)
        group2_matrix = np.vstack(shap_group2)

        t_stat, p_val = ttest_ind(group1_matrix, group2_matrix, axis=0, equal_var=False, nan_policy='omit')
        mean_diff = group1_matrix.mean(axis=0) - group2_matrix.mean(axis=0)

        print("[DEBUG] mean_diff:", mean_diff)
        print("[DEBUG] features length:", len(features))

        df_diff = pd.DataFrame({
            "feature": features,
            "mean_diff": mean_diff,
            "p_val": p_val
        }).sort_values("p_val")

        print("[DEBUG] df_diff shape:", df_diff.shape)

        if top_n is not None:
            df_diff = df_diff.head(top_n)

        plt.figure(figsize=figsize)
        sns.barplot(
            x=df_diff["mean_diff"],
            y=df_diff["feature"],
            color="steelblue"
        )
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if x_lim is not None:
            plt.xlim(x_lim)
        plt.tight_layout()
        plt.show()
    return (plot_differential_B6J_DO_shap_mean_abs_predictions,)


@app.cell
def _(plot_differential_B6J_DO_shap_mean_abs_predictions):
    plot_differential_B6J_DO_shap_mean_abs_predictions("combined_1126", "unsupervised", "age", top_n=20)
    return


@app.cell
def _():
    # plot_shap_summary_dot("geroscience_492", "unsupervised", "fll", top_n=20)
    return


@app.cell
def _():
    # plot_shap_mean_abs_prediction("combined_1126", "unsupervised", "age", top_n=20)
    return


@app.cell
def _():
    # plot_shap_mean_abs_prediction("combined_1126", "unsupervised", "fi", top_n=20, figsize=(10, 6), x_lim=(0.0, 0.7))
    return


@app.cell
def _():
    # plot_foldwise_errors("age", mae_, figsize=(6, 6))
    return


@app.cell
def _():
    # plot_foldwise_errors("fi", mae_, figsize=(6, 6))
    return


@app.cell
def _(data):
    data["combined_1126"]["features"].query('strain != "B6"')["strain"]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
