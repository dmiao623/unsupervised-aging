import marimo

__generated_with = "0.16.2"
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
    from scipy.stats import rankdata, ttest_ind
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
        ttest_ind,
    )


@app.cell
def _(Path, json, os, pd):
    unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])
    data_info_path         = unsupervised_aging_dir / "data/data_info_resample-test.json"

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
def _(data, pd):
    data["combined_1126"]["results"] = pd.read_csv("/projects/kumar-lab/miaod/projects/unsupervised-aging/data/model_evaluation_results/2025-10-08_model-evaluation-results__combined_1126__2025-09-20_kpms-v5_150_1_1-2-3-4-5-6-_old_1-_young_1.csv")
    print("reading json")
    return


@app.cell
def _(data, json):
    with open("/projects/kumar-lab/miaod/projects/unsupervised-aging/data/feature_matrices/2025-10-06_xcats__combined_1126__2025-09-20_kpms-v5_150_1_1-2-3-4-5-6-_old_1-_young-1.json", "r") as json_f:
        data["combined_1126"]["xcats"] = json.load(json_f)
    return


@app.cell
def _(Callable, Tuple, mean_absolute_error, mean_squared_error, pd, r2_score):
    Metric_t = Tuple[Callable[[pd.Series, pd.Series], float], str, bool]

    mae_  = (mean_absolute_error, "MAE", True)
    rmse_ = (mean_squared_error, "RMSE", True)
    r2_   = (r2_score, "R²", False)
    return Metric_t, mae_


@app.cell(hide_code=True)
def _(data, np, pd, plt, sns):
    def plot_foldwise_errors(
        y_cat,
        metric,
        *,
        model=None,
        figsize=(10, 6),
        x_cats=("unsupervised", "supervised", "all"),
        palette="Set1",
        title="",
        y_label="Error",
        x_label="Group",
        repeat_col="repeat",
        rename_map=None,
        legend=True,
        legend_inside=False,
    ):
        plot_data = []
        metric_fn, metric_name, is_loss = metric

        for label, dcs in zip(("B6J", "DO", "B6J/DO"), data.values()):
            results_all = dcs["results"].query('split == "test" and y_cat == @y_cat')
            for X_cat in x_cats:
                if model is None:
                    best_model, best_val = None, None
                    for m in results_all["model"].unique():
                        vals = []
                        results_model = results_all.query('model == @m and X_cat == @X_cat')
                        for fold in results_model["fold"].unique():
                            fold_subset = results_model[results_model["fold"] == fold]
                            for rep in fold_subset[repeat_col].unique():
                                fr = fold_subset[fold_subset[repeat_col] == rep]
                                vals.append(metric_fn(fr["y_true"], fr["y_pred"]))
                        if vals:
                            med = np.median(vals)
                            if best_val is None or (med < best_val if is_loss else med > best_val):
                                best_val, best_model = med, m
                    results = results_all.query('model == @best_model and X_cat == @X_cat')
                else:
                    results = results_all.query('model == @model and X_cat == @X_cat')

                for fold in results["fold"].unique():
                    fold_subset = results[results["fold"] == fold]
                    for rep in fold_subset[repeat_col].unique():
                        fr = fold_subset[fold_subset[repeat_col] == rep]
                        err = metric_fn(fr["y_true"], fr["y_pred"])
                        plot_data.append({y_label: err, x_label: label, "X_cat": X_cat})

        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=figsize)
        ax = sns.boxplot(
            data=df_plot,
            x=x_label,
            y=y_label,
            hue="X_cat",
            palette=palette,
            flierprops={"marker": "."},
        )
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            labels = [rename_map.get(l, l) if rename_map else l for l in labels]
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if legend_inside:
                ax.legend(handles, labels, loc="upper left")
            else:
                ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(Metric_t, Optional, Tuple, data, np, pd, plt, sns):
    def plot_single_boxplot_by_xcat(
        metric: Metric_t,
        y_cat: str,
        *,
        model: Optional[str]     = None,
        figsize: Tuple[int, int] = (10, 6),
        x_cats: Tuple[str, ...]  = ("unsupervised", "supervised", "all"),
        palette: str             = "Set1",
        title: str               = "",
        y_label: str             = "Error",
        x_label: str             = "Input Type",
        show_x_labels: bool      = True,
        repeat_col: str          = "repeat",
    ) -> None:
        plot_data = []
        metric_fn, metric_name, is_loss = metric
        results_all = data["combined_1126"]["results"].query(f'split == "test" and y_cat == @y_cat')

        for X_cat in x_cats:
            if model is None:
                best_model, best_val = None, None
                for m in results_all["model"].unique():
                    vals = []
                    rm = results_all.query('model == @m and X_cat == @X_cat')
                    for fold in rm["fold"].unique():
                        fs = rm[rm["fold"] == fold]
                        for rep in fs[repeat_col].unique():
                            fr = fs[fs[repeat_col] == rep]
                            vals.append(metric_fn(fr["y_true"], fr["y_pred"]))
                    if vals:
                        med = np.median(vals)
                        if best_val is None or (med < best_val if is_loss else med > best_val):
                            best_val, best_model = med, m
                results = results_all.query('model == @best_model and X_cat == @X_cat')
            else:
                results = results_all.query('model == @model and X_cat == @X_cat')

            for fold in results["fold"].unique():
                fs = results[results["fold"] == fold]
                for rep in fs[repeat_col].unique():
                    fr = fs[fs[repeat_col] == rep]
                    plot_data.append({y_label: metric_fn(fr["y_true"], fr["y_pred"]), x_label: X_cat})

        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=figsize, dpi=300)
        ax = sns.boxplot(data=df_plot, x=x_label, y=y_label, palette=palette)
        if not show_x_labels:
            plt.xticks([])
            plt.xlabel("")
        else:
            plt.xticks(rotation=90)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    return (plot_single_boxplot_by_xcat,)


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(Optional, Tuple, XGBRegressor, data, np, pd, plt, rankdata, shap):
    def plot_shap_summary_dot(
        dataset_label: str,
        X_cat: str,
        y_cat: str,
        *,
        top_n: Optional[int]                 = None,
        feat_prefix: str                     = "",
        model: str                           = "XGBoost",
        figsize: Tuple[int, int]             = (8, 6),
        title: str                           = "",
        y_label: str                         = "Feature",
        x_label: str                         = "SHAP value",
        x_lim: Optional[Tuple[float, float]] = None,
        cbar_label: str                      = "Feature Value Percentile",
        font_size: int                       = 10
    ) -> None:
        results = data[dataset_label]["results"].query(
            'split == "test" and model == @model and X_cat == @X_cat and y_cat == @y_cat and repeat == 1'
        )
        features = data[dataset_label]["xcats"][X_cat]
        type_map = {
            "n_estimators": int, "max_depth": int, "learning_rate": float, "subsample": float,
            "colsample_bytree": float, "min_child_weight": float, "gamma": float,
            "reg_alpha": float, "reg_lambda": float
        }
        num_folds = max(results["fold"]) + 1
        all_shap, all_X = [], []
        for fold in range(num_folds):
            train_mask, test_mask = results["fold"] != fold, results["fold"] == fold
            X_train, y_train = results.loc[train_mask, features], results.loc[train_mask, "y_true"]
            X_test = results.loc[test_mask, features]
            hp_row = results.loc[test_mask].iloc[0]
            params = {k.replace("model__", ""): type_map.get(k.replace("model__", ""), lambda x: x)(v)
                      for k, v in hp_row.items() if k.startswith("model__")}
            mdl = XGBRegressor(**params).fit(X_train, y_train)
            explainer = shap.Explainer(mdl)
            all_shap.append(explainer(X_test).values)
            all_X.append(X_test)
        shap_mat = np.vstack(all_shap)
        X_mat = pd.concat(all_X, axis=0)
        feat_names = features
        idxs = [i for i, f in enumerate(feat_names) if f.startswith(feat_prefix)]
        if not idxs:
            raise ValueError("No features match the provided prefix.")
        mean_abs = np.abs(shap_mat[:, idxs]).mean(0)
        order = np.argsort(mean_abs)[::-1]
        if top_n is not None:
            order = order[:top_n]
        feat_order = np.array(idxs)[order]
        cmap = plt.get_cmap("coolwarm")
        fig, ax = plt.subplots(figsize=figsize)
        for i, idx in enumerate(feat_order):
            sv, fv = shap_mat[:, idx], X_mat.iloc[:, idx]
            colors = cmap((rankdata(fv) - 1) / (len(fv) - 1 + 1e-8))
            jitter = np.random.uniform(-0.2, 0.2, size=len(sv))
            ax.scatter(sv, np.full_like(sv, i) + jitter, c=colors, s=10, alpha=0.7, edgecolors='none')
        yt = list(range(len(feat_order)))
        ax.set_yticks(yt[::-1])
        ax.set_yticklabels([feat_names[i] for i in feat_order][::-1], fontsize=font_size)
        ax.invert_yaxis()
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        if title:
            ax.set_title(title, fontsize=font_size + 1)
        if x_lim:
            ax.set_xlim(x_lim)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)
        ax.tick_params(axis='both', labelsize=font_size)
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(
    Optional,
    Tuple,
    XGBRegressor,
    data,
    mo,
    np,
    pd,
    plt,
    shap,
    sns,
    ttest_ind,
):
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
        x_label: str                         = "Δ SHAP value (B6 - DO)",
        x_lim: Optional[Tuple[float, float]] = None
    ) -> None:

        results = data[dataset_label]["results"].query(
            'split == "test" and model == @model and X_cat == @X_cat and y_cat == @y_cat'
        ).copy()

        results["strain"] = data[dataset_label]["features"].loc[results["sample_idx"], "strain"].values

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
    return


@app.cell
def _(data, mae_, plot_single_boxplot_by_xcat):
    filter1 = [
        "2025-09-20_kpms-v5_150_1",
        "2025-09-20_kpms-v5_150_2",
        "2025-09-20_kpms-v5_150_3",
        "2025-09-20_kpms-v5_150_4",
        "2025-09-20_kpms-v5_150_5",
        "2025-09-20_kpms-v5_150_6",
        "2025-09-20_kpms-v5_150__old_1",
        "2025-09-20_kpms-v5_150__young_1",
    ]
    x_cats1 = {k: v for k, v in data["combined_1126"]["xcats"].items() if k in filter1}
    plot_single_boxplot_by_xcat(mae_, "fi", figsize=(4,8), title="MAE of FI Pred.", show_x_labels=True, x_cats=filter1)
    return


@app.cell
def _(data, mae_, plot_single_boxplot_by_xcat):
    filter2 = [
        "2025-09-20_kpms-v5_150_4",
        "2025-09-20_kpms-v5_150_4__old",
        "2025-09-20_kpms-v5_150_5",
        "2025-09-20_kpms-v5_150_5__old",
        "2025-09-20_kpms-v5_150_6",
        "2025-09-20_kpms-v5_150_6__old",
    ]
    x_cats2 = {k: v for k, v in data["combined_1126"]["xcats"].items() if k in filter2}
    plot_single_boxplot_by_xcat(mae_, "fi", figsize=(4,8), title="MAE of FI Pred.", show_x_labels=True, x_cats=filter2)
    return


@app.cell
def _():
    # plot_single_boxplot_by_xcat(mae_, "fi", figsize=(4,8), title="MAE of FI Pred.", show_x_labels=True, _x_cats=data["combined_1126"]["xcats"])
    return


app._unparsable_cell(
    r"""
    -# plot_single_boxplot_by_xcat(mae_, figsize=(3, 6), title=\"MAE of PLL Prediction\", show_x_labels=True)
    """,
    name="_"
)


@app.cell
def _():
    # plot_differential_B6J_DO_shap_mean_abs_predictions("combined_1126", "unsupervised", "age", top_n=20)
    return


@app.cell
def _():
    # plot_shap_summary_dot("combined_1126", "unsupervised", "age", top_n=10, feat_prefix="syllable_frequency", figsize=(8, 6), title="")
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
    # plot_foldwise_errors("age", mae_, figsize=(5, 6), rename_map={
    #     "unsupervised": "unsupervised features",
    #     "supervised": "supervised features",
    #     "all": "combined features"
    # }, legend=False, y_label="Error (weeks)")
    return


@app.cell
def _():
    # plot_foldwise_errors("fi", mae_, figsize=(5, 6), rename_map={
    #     "unsupervised": "unsupervised features",
    #     "supervised": "supervised features",
    #     "all": "combined features"
    # }, legend=False, y_label="Error (FI)")
    return


@app.cell
def _():
    # subset = data["combined_1126"]["results"].query('split == "test" and y_cat == "fi" and X_cat == "unsupervised" and model == "XGBoost"')
    # errors = subset.groupby(["fold", "repeat"]).apply(lambda df: mean_absolute_error(df.y_true, df.y_pred)).values
    # mean_mae = errors.mean()
    # std_mae = errors.std(ddof=1)
    # print(f"{mean_mae:.3f} ± {std_mae:.3f}")

    # err_pct = (
    #     subset
    #       .groupby(["fold", "repeat"])
    #       .apply(lambda g: np.mean(np.abs(g.y_true - g.y_pred) / g.y_true) * 100)
    #       .values
    # )

    # print(f"Percentage error: {err_pct.mean():.2f}% ± {err_pct.std(ddof=1):.2f}%")
    return


@app.cell
def _():
    # def _foldwise_boxplot(
    #     y_cat,
    #     metric,
    #     *,
    #     ax,
    #     data,
    #     model=None,
    #     x_cats=("unsupervised", "supervised", "all"),
    #     palette="Set1",
    #     title="",
    #     y_label="Error",
    #     x_label="Group",
    #     repeat_col="repeat",
    #     rename_map=None,
    # ):
    #     metric_fn, _, is_loss = metric
    #     plot_data = []

    #     for label, dcs in zip(("B6J", "DO", "B6J/DO"), data.values()):
    #         if label != "DO":
    #             continue
    #         res_all = dcs["results"].query('split == "test" and y_cat == @y_cat')
    #         for X_cat in x_cats:
    #             if model is None:
    #                 best_model, best_val = None, None
    #                 for m in res_all["model"].unique():
    #                     vals = []
    #                     res_m = res_all.query('model == @m and X_cat == @X_cat')
    #                     for fold in res_m["fold"].unique():
    #                         fold_subset = res_m[res_m["fold"] == fold]
    #                         for rep in fold_subset[repeat_col].unique():
    #                             fr = fold_subset[fold_subset[repeat_col] == rep]
    #                             vals.append(metric_fn(fr["y_true"], fr["y_pred"]))
    #                     if vals:
    #                         med = np.median(vals)
    #                         if best_val is None or (med < best_val if is_loss else med > best_val):
    #                             best_val, best_model = med, m
    #                 res = res_all.query('model == @best_model and X_cat == @X_cat')
    #             else:
    #                 res = res_all.query('model == @model and X_cat == @X_cat')

    #             for fold in res["fold"].unique():
    #                 fold_subset = res[res["fold"] == fold]
    #                 for rep in fold_subset[repeat_col].unique():
    #                     fr = fold_subset[fold_subset[repeat_col] == rep]
    #                     err = metric_fn(fr["y_true"], fr["y_pred"])
    #                     plot_data.append({y_label: err, x_label: label, "X_cat": X_cat})

    #     df_plot = pd.DataFrame(plot_data)
    #     sns.boxplot(
    #         data=df_plot,
    #         x=x_label,
    #         y=y_label,
    #         hue="X_cat",
    #         palette=palette,
    #         flierprops={"marker": "."},
    #         ax=ax,
    #     )
    #     handles, labels = ax.get_legend_handles_labels()
    #     labels = [rename_map.get(l, l) if rename_map else l for l in labels]
    #     ax.get_legend().remove()
    #     ax.set_title(title)
    #     return handles, labels

    # _foldwise_boxplot()
    return


@app.cell
def _():
    # rename = {
    #     "unsupervised": "unsupervised features",
    #     "supervised": "supervised features",
    #     "all": "combined features",
    # }

    # plot_foldwise_errors_side_by_side(
    #     left_kwargs=dict(
    #         y_cat="age",
    #         metric=mae_,
    #         title="MAE of Age Prediction Models",
    #         model=None,  # or a specific model name
    #     ),
    #     right_kwargs=dict(
    #         y_cat="fi",
    #         metric=mae_,
    #         title="MAE of FI Prediction Models",
    #         model=None,
    #     ),
    #     data=data,  # your existing dict of datasets
    #     figsize_left=(3, 6),
    #     figsize_right=(3, 6),  # example of different size
    #     palette="Set1",
    #     rename_map=rename,
    # )
    return


@app.cell
def _():
    # def _plot_foldwise_errors(y_cat, metric, *, model=None, figsize=(10,6),
    #                          x_cats=("unsupervised","supervised","all"), palette="Set1",
    #                          title="", y_label="Error", x_label="Group",
    #                          repeat_col="repeat", rename_map=None):
    #     plot_data=[]
    #     metric_fn,_,is_loss=metric
    #     for label,dcs in zip(("B6J","DO","B6J/DO"),data.values()):
    #         if label!="DO": continue
    #         results_all=dcs["results"].query('split=="test" and y_cat==@y_cat')
    #         for X_cat in x_cats:
    #             if model is None:
    #                 best_model,best_val=None,1e9
    #                 for m in results_all["model"].unique():
    #                     vals=[]
    #                     res_m=results_all.query('model==@m and X_cat==@X_cat')
    #                     for fold in res_m["fold"].unique():
    #                         frs=res_m[res_m["fold"]==fold]
    #                         for rep in frs[repeat_col].unique():
    #                             fr=frs[frs[repeat_col]==rep]
    #                             vals.append(metric_fn(fr["y_true"],fr["y_pred"]))
    #                     if vals:
    #                         med=np.median(vals)
    #                         cond=med<best_val if is_loss else med>best_val if best_val is not None else True
    #                         if cond: best_val, best_model=med,m
    #                 if best_model is None: continue
    #                 results=results_all.query('model==@best_model and X_cat==@X_cat')
    #             else:
    #                 results=results_all.query('model==@model and X_cat==@X_cat')
    #             if results.empty: continue
    #             for fold in results["fold"].unique():
    #                 frs=results[results["fold"]==fold]
    #                 for rep in frs[repeat_col].unique():
    #                     fr=frs[frs[repeat_col]==rep]
    #                     err=metric_fn(fr["y_true"],fr["y_pred"])
    #                     plot_data.append({y_label:err,x_label:label,"X_cat":X_cat})
    #     if not plot_data:
    #         raise ValueError(f"No test-set data found for y_cat={y_cat} in DO")
    #     df_plot=pd.DataFrame(plot_data)
    #     fig,ax=plt.subplots(figsize=figsize)
    #     sns.boxplot(data=df_plot,x=x_label,y=y_label,hue="X_cat",
    #                 palette=palette,flierprops={"marker":"."},ax=ax)
    #     handles,labels=ax.get_legend_handles_labels()
    #     labels=[rename_map.get(l,l) if rename_map else l for l in labels]
    #     ax.legend_.remove()
    #     ax.set_title(title)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     fig.tight_layout()
    #     plt.show()
    #     return handles,labels

    # def legend_only(handles,labels,*,ncol=1,figsize=(3,2)):
    #     fig,ax=plt.subplots(figsize=figsize)
    #     ax.axis("off")
    #     ax.legend(handles,labels,loc="center",frameon=False,ncol=ncol)
    #     fig.tight_layout()
    #     plt.show()



    # _plot_foldwise_errors("fll", mae_, figsize=(2.5, 6), rename_map={
    #     "unsupervised": "unsupervised features",
    #     "supervised": "supervised features",
    #     "all": "combined features"
    # }, y_label="Error (PLL)")
    return


@app.cell
def _():
    # from matplotlib.patches import Patch

    # fig, ax = plt.subplots(figsize=(2.5, 1))
    # ax.axis('off')
    # ax.legend(
    #     handles=[
    #         Patch(color='red', label='unsupervised features'),
    #         Patch(color='blue', label='supervised features'),
    #         Patch(color='green', label='combined features')
    #     ],
    #     loc='center',
    #     frameon=False
    # )
    # plt.tight_layout()
    # plt.show()
    return


@app.cell
def _(data, np, pd):
    _metrics = [
        ("MAE",  lambda y, p: np.mean(np.abs(y - p))),
        ("RMSE", lambda y, p: np.sqrt(np.mean((y - p) ** 2))),
        ("R2",   lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)),
    ]

    _models = ["Random Forest", "XGBoost"]

    def _best_model(results_all, X_cat, repeat_col):
        sel_fn = _metrics[0][1]
        best_m, best_val = None, None
        for m in _models:
            vals = []
            rm = results_all.query('model == @m and X_cat == @X_cat')
            for f in rm["fold"].unique():
                fs = rm[rm["fold"] == f]
                for r in fs[repeat_col].unique():
                    fr = fs[fs[repeat_col] == r]
                    vals.append(sel_fn(fr["y_true"], fr["y_pred"]))
            if vals:
                med = np.median(vals)
                if best_val is None or med < best_val:
                    best_val, best_m = med, m
        return best_m

    def summarize_errors_with_err(
        y_cat,
        *,
        model=None,
        x_cats=("unsupervised", "supervised", "all"),
        repeat_col="repeat"
    ):
        central, err = [], []
        for X_cat in x_cats:
            row_c, row_e = {}, {}
            for label, dcs in zip(("B6J", "DO", "B6J/DO"), data.values()):
                ra = dcs["results"].query('split == "test" and y_cat == @y_cat')
                m_use = model or _best_model(ra, X_cat, repeat_col)
                res = ra.query('model == @m_use and X_cat == @X_cat')
                vals = {m: [] for m, _ in _metrics}
                for f in res["fold"].unique():
                    fs = res[res["fold"] == f]
                    for r in fs[repeat_col].unique():
                        fr = fs[fs[repeat_col] == r]
                        y, p = fr["y_true"].values, fr["y_pred"].values
                        for mname, mfn in _metrics:
                            vals[mname].append(mfn(y, p))
                for mname in vals:
                    v = vals[mname]
                    row_c[f"{label}_{mname}"] = np.mean(v) if v else np.nan
                    row_e[f"{label}_{mname}_err"] = np.std(v, ddof=1) if len(v) > 1 else np.nan
            central.append(row_c)
            err.append(row_e)
        return pd.DataFrame(central, index=x_cats), pd.DataFrame(err, index=x_cats)

    def _sig(x, n=5):
        return f"{x:.{n}g}"

    def merge_mean_err(mean_df, err_df, n=5):
        out = mean_df.copy()
        for col in mean_df.columns:
            out[col] = [
                f"{_sig(mean_df[col].iloc[i], n)} ± {_sig(err_df[f'{col}_err'].iloc[i], n)}"
                for i in range(len(mean_df))
            ]
        return out.astype(str)

    age_df, age_err = summarize_errors_with_err("age")
    fi_df,  fi_err  = summarize_errors_with_err("fi")

    age_tbl = merge_mean_err(age_df, age_err)
    fi_tbl  = merge_mean_err(fi_df,  fi_err)

    age_tbl, fi_tbl
    return


@app.cell
def _(data, np, pd):
    _metrics = [
        ("MAE",  lambda y, p: np.mean(np.abs(y - p))),
        ("RMSE", lambda y, p: np.sqrt(np.mean((y - p) ** 2))),
        ("R2",   lambda y, p: 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)),
    ]


    def _best_model_do(res_all,X_cat,repeat_col):
        mf=_metrics[0][1];b_m,b_v=None,None
        for m in res_all["model"].unique():
            v=[];rm=res_all.query('model==@m and X_cat==@X_cat')
            for f in rm["fold"].unique():
                fs=rm[rm["fold"]==f]
                for r in fs[repeat_col].unique():
                    fr=fs[fs[repeat_col]==r]
                    v.append(mf(fr["y_true"],fr["y_pred"]))
            if v:
                med=np.median(v)
                if b_v is None or med<b_v:
                    b_v,b_m=med,m
        return b_m

    def summarize_errors_do(y_cat,x_cats=("unsupervised","supervised","all"),repeat_col="repeat",model=None):
        cent,err=[],[]
        for X_cat in x_cats:
            rc,re={},{}
            for lbl,dcs in zip(("B6J","DO","B6J/DO"),data.values()):
                if lbl!="DO":continue
                ra=dcs["results"].query('split=="test" and y_cat==@y_cat')
                m_use=model or _best_model_do(ra,X_cat,repeat_col)
                res=ra.query('model==@m_use and X_cat==@X_cat')
                vals={n:[] for n,_ in _metrics}
                for f in res["fold"].unique():
                    fs=res[res["fold"]==f]
                    for r in fs[repeat_col].unique():
                        fr=fs[fs[repeat_col]==r]
                        y,p=fr["y_true"].values,fr["y_pred"].values
                        for n,fm in _metrics: vals[n].append(fm(y,p))
                for n in vals:
                    v=vals[n]
                    rc[f"DO_{n}"]=np.mean(v) if v else np.nan
                    re[f"DO_{n}_err"]=np.std(v,ddof=1) if len(v)>1 else np.nan
            cent.append(rc);err.append(re)
        return pd.DataFrame(cent,index=x_cats),pd.DataFrame(err,index=x_cats)

    def _sig(x,n=5):return f"{x:.{n}g}"

    def _merge_mean_err(mean_df,err_df,n=5):
        out=mean_df.copy()
        for c in mean_df.columns:
            out[c]=[f"{_sig(mean_df[c].iloc[i],n)} ± {_sig(err_df[f'{c}_err'].iloc[i],n)}"
                    for i in range(len(mean_df))]
        return out.astype(str)

    fll_df,fll_err=summarize_errors_do("fll")
    fll_tbl=_merge_mean_err(fll_df,fll_err)
    fll_tbl
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
