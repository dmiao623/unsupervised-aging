import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt 
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns
    from pathlib import Path
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Sequence
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
        mpl,
        np,
        os,
        pd,
        plt,
        r2_score,
        sns,
    )


@app.cell
def _(Path, json, os, pd):
    feature_df_path = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices/c_final.csv"
    feature_json_path = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices/c_final.json"

    feature_df = pd.read_csv(feature_df_path)
    with feature_json_path.open("r") as f:
        feature_json = json.load(f)
    return feature_df, feature_json


@app.cell
def _():
    # _v = feature_json.pop("2025-09-20_kpms-v5_150_6")
    # feature_json["unsupervised"] = _v
    # # feature_json
    # result_df.replace()
    return


@app.cell
def _(Path, os, pd):
    result_df_path = Path(os.environ["UNSUPERVISED_AGING"]) / "data/model_evaluation_results/c_final_res.csv"

    result_df = pd.read_csv(result_df_path)
    return (result_df,)


@app.cell
def _(mo):
    mo.md(r"""## Data Distribution""")
    return


@app.cell
def _(feature_df):
    for strain in ("B6", "DO"):
        strain_feature_df = feature_df.query("strain == @strain")
        print(strain, {
            "num_mice": len(set(strain_feature_df["mouse_id"])),
            "num_tests": len(strain_feature_df),    
        })
    return


@app.cell(hide_code=True)
def _():
    # def _plot_diet_distribution(
    #     *,
    #     diet_colors = {
    #         "AL": "gray", "1D": "lightblue", "2D": "blue",
    #         "20": "orange", "40": "red"
    #     },
    #     figsize = (3, 6),
    #     fontsize = 10,
    # ):

    #     combined = pd.DataFrame({
    #         "B6J": feature_df.query("strain == 'B6'")["diet"].value_counts(),
    #         "DO": feature_df.query("strain == 'DO'")["diet"].value_counts()
    #     }).fillna(0)

    #     combined = combined.loc[combined.index.intersection(diet_colors.keys())]
    #     combined = combined.loc[sorted(combined.index, key=lambda x: list(diet_colors).index(x))]

    #     plt.figure(figsize=figsize, dpi=300)
    #     ax = combined.T.plot(
    #         kind="bar",
    #         stacked=True,
    #         color=[diet_colors[diet] for diet in combined.index],
    #         ax=plt.gca()
    #     )
    #     ax.legend_.remove()
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)

    #     ax.set_ylabel("Count", fontsize=fontsize)
    #     ax.set_xlabel("Strain", fontsize=fontsize)
    #     ax.tick_params(axis="both", labelsize=fontsize)

    #     plt.tight_layout()
    #     plt.show()

    # _plot_diet_distribution()
    return


@app.cell(hide_code=True)
def _():
    # def _plot_age_fi_scatter(
    #     *,
    #     diet_colors={"AL": "gray", "1D": "lightblue", "2D": "blue", "20": "orange", "40": "red"},
    #     figsize=(7, 5),
    #     fontsize=10,
    # ):
    #     df = feature_df.copy()
    #     strain_markers = {"B6": "o", "DO": "^"}

    #     plt.figure(figsize=figsize, dpi=300)
    #     labeled_diets = set()

    #     for strain in df.strain.unique():
    #         for diet in df.diet.unique():
    #             subset = df[(df.strain == strain) & (df.diet == diet)]
    #             label = diet if diet not in labeled_diets else None
    #             if label:
    #                 labeled_diets.add(diet)
    #             plt.scatter(
    #                 subset.age,
    #                 subset.fi,
    #                 label=label,
    #                 c=diet_colors[diet],
    #                 marker=strain_markers[strain],
    #                 alpha=0.6,
    #                 edgecolors="w",
    #                 s=60,
    #             )

    #     df["age_bin"] = pd.cut(df.age, bins=[0, 50, 100, 150, 200])
    #     summary = (
    #         df.groupby(["strain", "age_bin"]).fi.agg(["mean", "std"]).reset_index()
    #     )
    #     summary["age_mid"] = summary.age_bin.map(lambda x: x.mid)

    #     for strain in summary.strain.unique():
    #         sub = summary[summary.strain == strain].sort_values("age_mid")
    #         plt.errorbar(
    #             sub.age_mid,
    #             sub["mean"],
    #             yerr=sub["std"].fillna(0),
    #             fmt="-o" if strain == "B6" else "-^",
    #             color="black",
    #             linewidth=2,
    #             capsize=3,
    #         )

    #     diet_handles = [mpl.patches.Patch(facecolor=c, label=d) for d, c in diet_colors.items()]
    #     strain_handles = [
    #         mpl.lines.Line2D([0], [0], marker=m, color="w", label=s, markerfacecolor="black", markersize=10)
    #         for s, m in strain_markers.items()
    #     ]

    #     first = plt.legend(handles=diet_handles, title="Diet", bbox_to_anchor=(1.05, 0.3), loc="center left", fontsize=fontsize, title_fontsize=fontsize)
    #     plt.gca().add_artist(first)
    #     plt.legend(handles=strain_handles, title="Strain", bbox_to_anchor=(1.05, 0.9), loc="upper left", fontsize=fontsize, title_fontsize=fontsize)

    #     plt.xlabel("Age (weeks)", fontsize=fontsize)
    #     plt.ylabel("Score (CFI)", fontsize=fontsize)
    #     ax = plt.gca()
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["top"].set_visible(False)
    #     ax.tick_params(axis="both", labelsize=fontsize)
    #     plt.tight_layout()
    #     plt.show()


    # _plot_age_fi_scatter(fontsize=10)
    return


@app.cell(hide_code=True)
def _(feature_df, mpl, pd, plt):
    def plot_combined_diet_age_fi(
        *,
        diet_colors = {
            "AL": "gray", "1D": "lightblue", "2D": "blue",
            "20": "orange", "40": "red"
        },
        figsize = (10, 6),
        wspace = 0.3, 
    ):
        combined = pd.DataFrame({
            "B6J": feature_df.query("strain == 'B6'")["diet"].value_counts(),
            "DO": feature_df.query("strain == 'DO'")["diet"].value_counts()
        }).fillna(0)

        combined = combined.loc[combined.index.intersection(diet_colors.keys())]
        combined = combined.loc[sorted(combined.index, key=lambda x: list(diet_colors).index(x))]

        # Prepare scatter data
        df = feature_df.copy()
        strain_markers = {
            "B6": "o",
            "DO": "^",
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300, gridspec_kw={'width_ratios': [1, 2]})
        fig.subplots_adjust(wspace=wspace)

        combined.T.plot(
            kind="bar",
            stacked=True,
            color=[diet_colors[diet] for diet in combined.index],
            ax=ax1
        )
        ax1.legend_.remove()
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Strain")
        ax1.set_title("Diet Distribution by Strain")

        labeled_diets = set()
        for strain in df["strain"].unique():
            for diet in df["diet"].unique():
                subset = df[(df["strain"] == strain) & (df["diet"] == diet)]
                label = diet if diet not in labeled_diets else None
                if label:
                    labeled_diets.add(diet)
                ax2.scatter(
                    subset["age"], subset["fi"],
                    label=label,
                    c=diet_colors[diet],
                    marker=strain_markers.get(strain, 'o'),
                    alpha=0.6,
                    edgecolors="w",
                    s=60
                )

        df["age_bin"] = pd.cut(df["age"], bins=[0, 50, 100, 150, 200])
        summary = df.groupby(["strain", "age_bin"])["fi"].agg(["mean", "sem"]).reset_index()
        summary["age_mid"] = summary["age_bin"].apply(lambda x: x.mid)

        for strain in summary["strain"].unique():
            sub = summary[summary["strain"] == strain]
            ax2.errorbar(
                sub["age_mid"], sub["mean"], yerr=sub["sem"],
                fmt="-o" if strain == "B6" else "-^",
                color="black",
                linewidth=2,
                capsize=3,
                label=None
            )

        ax2.set_xlabel("Age (in weeks)")
        ax2.set_ylabel("Score (CFI)")
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_title("Age vs FI Scatter by Diet and Strain")

        diet_legend = [mpl.patches.Patch(facecolor=color, label=diet) for diet, color in diet_colors.items()]
        strain_legend = [mpl.lines.Line2D([0], [0], marker=marker, color='w',
                                label=strain, markerfacecolor='black', markersize=10)
                         for strain, marker in strain_markers.items()]

        first_legend = ax2.legend(handles=diet_legend, title="Diet",
                                  bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax2.add_artist(first_legend)

        ax2.legend(handles=strain_legend, title="Strain",
                   bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.show()

    plot_combined_diet_age_fi(wspace=0.5)
    return


@app.cell
def _(mo):
    mo.md(r"""## Model Evaluation Results""")
    return


@app.cell
def _(Callable, Tuple, mean_absolute_error, mean_squared_error, pd, r2_score):
    Metric_t = Tuple[Callable[[pd.Series, pd.Series], float], str, bool]

    mae_  = (mean_absolute_error, "MAE", True)
    rmse_ = (mean_squared_error, "RMSE", True)
    r2_   = (r2_score, "R²", False)
    return Metric_t, mae_


@app.cell
def _(feature_json):
    feature_json
    return


@app.cell
def _(Metric_t, Optional, Tuple, mae_, np, pd, plt, result_df, sns):
    def plot_single_boxplot_by_xcat(
        metric: Metric_t,
        y_cat: str,
        *,
        model: Optional[str]     = None,
        figsize: Tuple[int, int] = (3, 6),
        x_cats: Tuple[str, ...]  = ("2025-09-20_kpms-v5_150_6", "supervised"),
        palette: str             = "Set1",
        title: str               = "",
        y_label: str             = "Error",
        x_label: str             = "Input Type",
        show_x_labels: bool      = True,
        repeat_col: str          = "repeat",
        X_label_remap            = {"2025-09-20_kpms-v5_150_6": "unsupervised", "supervised": "supervised"},
    ) -> None:
        plot_data = []
        metric_fn, metric_name, is_loss = metric
        results_all = result_df.query(f'split == "test" and y_cat == @y_cat')

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
                    plot_data.append({y_label: metric_fn(fr["y_true"], fr["y_pred"]), x_label: X_label_remap[X_cat]})

        df_plot = pd.DataFrame(plot_data)

        stats = df_plot.groupby(x_label)[y_label].agg(Mean='mean', Std='std').reset_index()
        print(stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    
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

    plot_single_boxplot_by_xcat(mae_, "age", title="MAE of Age Prediction Models")
    return (plot_single_boxplot_by_xcat,)


@app.cell
def _(mae_, plot_single_boxplot_by_xcat):
    plot_single_boxplot_by_xcat(mae_, "fi", title="MAE of FI Prediction Models")
    return


@app.cell
def _(mo):
    mo.md(r"""## Differential Analysis""")
    return


@app.cell
def _(feature_df):
    b6j_df, do_df = feature_df.query("strain == 'B6'"), feature_df.query("strain == 'DO'")

    q25 = feature_df["age"].quantile(0.25)
    q75 = feature_df["age"].quantile(0.75)

    young_df = feature_df[feature_df["age"] <= q25]
    old_df = feature_df[feature_df["age"] >= q75]
    def get_syllable_stats(df):
        cols = [c for c in df.columns if c.startswith("total_duration")]
        return {c: (df[c].mean(), df[c].std()) for c in cols}

    b6j_stats = get_syllable_stats(b6j_df)
    do_stats = get_syllable_stats(do_df)
    young_stats = get_syllable_stats(young_df)
    old_stats = get_syllable_stats(old_df)
    return b6j_stats, do_stats, old_stats, young_stats


@app.cell
def _(b6j_stats, do_stats, np, plt):
    def plot_normalized_syllable_usage(baseline_means, other_means, baseline_label="B6J", other_label="DO", figsize=(10,5), xtick_step=5, show=True, log_scale=False, colors=None):
        keys = [k for k in baseline_means if k in other_means]
        b = np.array([baseline_means[k] for k in keys], dtype=float)
        o = np.array([other_means[k] for k in keys], dtype=float)
        norm = np.divide(o, b, out=np.full_like(o, np.nan), where=b!=0)
        x = np.arange(1, len(keys)+1)
        fig, ax = plt.subplots(figsize=figsize)
        if colors is not None:
            if isinstance(colors, str):
                cm = plt.get_cmap(colors)
                cols = cm(np.linspace(0.25, 0.75, 2))
            else:
                cols = list(colors)[:2]
            ax.set_prop_cycle(color=cols)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(x, np.ones_like(x), label=f"{baseline_label} (baseline)")
        ax.plot(x, norm, label=f"{other_label} / {baseline_label}", marker="o")
        if xtick_step:
            ticks = list(range(1, len(keys)+1, xtick_step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(i) for i in ticks])
        ax.set_xlabel("Syllable Number")
        ax.set_ylabel("Normalized syllable usage")
        ax.set_title(f"Normalized syllable frequency: {other_label} vs {baseline_label}")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper left")
        fig.tight_layout()
        if show:
            plt.show()



    b6j_means = {c: v[0] for c, v in b6j_stats.items()}
    do_means = {c: v[0] for c, v in do_stats.items()}
    plot_normalized_syllable_usage(b6j_means, do_means, show=True, log_scale=True)
    return (plot_normalized_syllable_usage,)


@app.cell
def _(old_stats, plot_normalized_syllable_usage, young_stats):
    young_means = {c: v[0] for c, v in young_stats.items()}
    old_means = {c: v[0] for c, v in old_stats.items()}
    plot_normalized_syllable_usage(young_means, old_means, show=True, baseline_label="Young", other_label="Old", log_scale=True, colors="viridis")
    return


@app.cell
def _():
    import shap
    return (shap,)


@app.cell
def _(
    Optional,
    Tuple,
    XGBRegressor,
    feature_json,
    mo,
    np,
    pd,
    plt,
    result_df,
    shap,
    sns,
):
    def plot_shap_mean_abs_prediction(
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
    ):
        results = result_df.query(
            'split == "test" and model == @model and X_cat == @X_cat and y_cat == @y_cat and repeat == 0'
        )
        features = feature_json[X_cat]
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
        return feature_importance.to_dict()


    preds = plot_shap_mean_abs_prediction("2025-09-20_kpms-v5_150_6", "age");
    return (preds,)


@app.cell
def _(preds):
    pred_by_syllb = {}
    for i in range(92):
        pred_by_syllb[i] = (
            preds.get(f"avg_bout_length_{i}") + 
            preds.get(f"total_duration_{i}") + 
            preds.get(f"num_bouts_{i}")
        )
    pred_by_syllb
    return (pred_by_syllb,)


@app.cell
def _(pred_by_syllb):
    print(sorted(pred_by_syllb, key=pred_by_syllb.get, reverse=True)[:10])
    return


@app.cell
def _(np, plt, pred_by_syllb):
    def plot_pred_by_syllable(pred_by_syllb, label="SHAP value", figsize=(10,5), xtick_step=5, show=True, log_scale=False, color=None):
        x = np.array(sorted(pred_by_syllb.keys()))
        y = np.array([pred_by_syllb[k] for k in x], dtype=float)
        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(x, y, marker="o", label=label, color=color)
        if xtick_step:
            ticks = x[::xtick_step]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(t)) for t in ticks])
        ax.set_xlabel("Syllable Number")
        ax.set_ylabel("Value")
        ax.set_title("SHAP value by syllable")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper left")
        fig.tight_layout()
        if show:
            plt.show()

    plot_pred_by_syllable(pred_by_syllb, label="SHAP score", xtick_step=5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
