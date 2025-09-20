import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import pickle
    import scipy.cluster.hierarchy as sch
    import seaborn as sns

    from pathlib import Path
    from scipy.spatial.distance import squareform
    from scipy.stats import linregress, pearsonr
    from typing import Mapping, Optional, Sequence, Tuple

    mpl.style.use("default")
    return (
        Mapping,
        Optional,
        Path,
        Sequence,
        Tuple,
        json,
        linregress,
        np,
        os,
        pd,
        pearsonr,
        plt,
        sch,
        sns,
        squareform,
    )


@app.cell
def _(Path, os):
    features_df_path = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices/2025-07-16_kpms-v3-supervised_feature-matrix.csv"
    xcats_path       = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices/2025-07-16_kpms-v3-supervised_xcats.json"
    results_df_path  = Path(os.environ["UNSUPERVISED_AGING"]) / "data/model_evaluation_results/2025-07-18_kpms-v3-supervised_results.csv"
    return features_df_path, results_df_path, xcats_path


@app.cell
def _(features_df_path, json, pd, results_df_path, xcats_path):
    features_df = pd.read_csv(features_df_path)
    results_df = pd.read_csv(results_df_path, low_memory=False)
    with xcats_path.open("r") as f:
        xcats = json.load(f)
    xcats.keys()
    return features_df, results_df, xcats


@app.cell
def _(Sequence, features_df, pd, plt, sns, xcats):
    def plot_horizontal_heatmap(
        df: pd.DataFrame,
        *,
        targets: Sequence[str] = ["age", "fi"],
        th: float              = 0.2
    ):
        feature_set = [col for col in df.columns if col not in targets]
        corr_df = pd.DataFrame(
            {tgt: df[feature_set + [tgt]].corr().loc[tgt, feature_set] for tgt in targets}
        ).T
        mask = corr_df.abs().gt(th).any(axis=0)
        filtered_df = corr_df.loc[:, mask]
        if filtered_df.empty:
            raise ValueError(f"No features exceed |r| > {th:.2f}.")
        fig_h = 1.5 * len(targets)
        fig_w = 0.6 * filtered_df.shape[1]
        fig, axes = plt.subplots(len(targets), 1, figsize=(fig_w, fig_h), sharex=True)
        if len(targets) == 1:
            axes = [axes]
        for ax, tgt in zip(axes, targets):
            sns.heatmap(
                filtered_df.loc[[tgt]],
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                annot=True,
                fmt=".2f",
                cbar=False,
                ax=ax,
            )
            ax.set_ylabel(tgt, rotation=0, fontsize=12, ha="right", va="center", labelpad=50)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
            plt.setp(ax.get_xticklabels(), ha="right")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

    plot_horizontal_heatmap(features_df[["age", "fi"] + xcats["kpms-v2_all"]])
    return


@app.cell
def _(Mapping, Sequence, Tuple, features_df, pd, plt, sns, xcats):
    def plot_correlation_scatterplot(
        df: pd.DataFrame,
        labels: Mapping[str, Sequence[str]],
        *,
        targets: Sequence[str]   = ["age", "fi"],
        th: float                = 0.,
        lim: Tuple[float, float] = (-1., 1.),
    ):
        if len(targets) != 2:
            raise ValueError("targets must contain exactly two elements.")
        feature_set = [col for col in df.columns if col not in targets]
        corr_x = df[feature_set + [targets[0]]].corr().loc[targets[0], feature_set]
        corr_y = df[feature_set + [targets[1]]].corr().loc[targets[1], feature_set]
        corr_df = pd.DataFrame({"feature": feature_set, "x": corr_x.values, "y": corr_y.values})
        corr_df = corr_df[corr_df[["x", "y"]].abs().gt(th).any(axis=1)]
        if corr_df.empty:
            raise ValueError(f"No features exceed |r| > {th:.2f}.")
        def assign_label(col):
            for label, cols in labels.items():
                if col in cols:
                    return label
            return "Other"
        corr_df["label"] = corr_df["feature"].apply(assign_label)
        unique_labels = corr_df["label"].unique()
        palette = dict(zip(unique_labels, sns.color_palette(n_colors=len(unique_labels))))
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=corr_df, x="x", y="y", hue="label", palette=palette, edgecolor="black")
        plt.axhline(0, linewidth=0.5)
        plt.axvline(0, linewidth=0.5)
        plt.xlim(*lim)
        plt.ylim(*lim)
        plt.xlabel(f"Correlation with {targets[0]}")
        plt.ylabel(f"Correlation with {targets[1]}")
        plt.legend(title="Label")
        plt.tight_layout()
        plt.show()

    unsupervised_labels = {
        "latent_embedding": [col for col in xcats["kpms-v2_all"] if col.startswith("latent_embedding")],
        "syllable_frequency": [col for col in xcats["kpms-v2_all"] if col.startswith("syllable_frequency")],
        "metasyllable_frequency": [col for col in xcats["kpms-v2_all"] if col.startswith("metasyllable_frequency")],
        "transition_matrix": [col for col in xcats["kpms-v2_all"] if col.startswith("transition_matrix")],
    }
    plot_correlation_scatterplot(
        features_df[["age", "fi"] + xcats["kpms-v2_all"] + xcats["supervised"]], 
        {**unsupervised_labels, "supervised": xcats["supervised"]},
        lim=(-0.75, 0.75)
    )
    return


@app.cell
def _(pd, pearsonr, plt, sns):
    def plot_scatterplot_of_features(
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ):
        data = df[[x_col, y_col]].dropna()
        r, _ = pearsonr(data[x_col], data[y_col])

        plt.figure(figsize=(5, 4))
        sns.scatterplot(x=x_col, y=y_col, data=data, s=35)

        sns.regplot(
            x=x_col, y=y_col, data=data,
            scatter=False, line_kws={"color": "black", "linewidth": 1}
        )

        plt.title(f"{y_col} vs {x_col}  (r = {r:.2f})")
        plt.tight_layout()
        plt.show()

    # plot_scatterplot_of_features(features_df, "transition_matrix_stationary_unknown", "age")
    # plot_scatterplot_of_features(features_df, "transition_matrix_stationary_unknown", "fi")
    return


@app.cell
def _(features_df, np, pd, plt, sch, sns, squareform, xcats):
    def plot_correlation_matrix_heatmap(
        df: pd.DataFrame,
        *,
        cluster:bool = True
    ):
        X = (
            df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        )

        X = X.loc[:, X.nunique(dropna=True) > 1]

        if X.shape[1] < 2:
            raise ValueError("Need at least two valid features to plot a grid.")

        corr = X.corr(method="pearson")
        var  = X.var()
        np.fill_diagonal(corr.values, var)

        if cluster:
            dist_for_linkage = 1 - corr.abs().fillna(0).values
            linkage = sch.linkage(
                squareform(dist_for_linkage, checks=False),
                method="average",
                optimal_ordering=True,
            )
            order = sch.dendrogram(linkage, no_plot=True)["leaves"]
            corr  = corr.iloc[order, order]

        n = len(corr)
        fig_size = max(4, n * 0.35)
        plt.figure(figsize=(fig_size, fig_size))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            vmin=-1, vmax=1,
            linewidths=0.4,
            linecolor="black",
            square=True,
            cbar_kws={"label": "Pearson r (off-diag) / Variance (diag)"},
        )

        plt.xticks(rotation=90, ha="center", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()

    plot_correlation_matrix_heatmap(features_df[xcats["kpms-v2_all"]])
    return


@app.cell
def _(
    Optional,
    Sequence,
    features_df,
    np,
    pd,
    plt,
    sch,
    sns,
    squareform,
    xcats,
):
    def plot_cross_correlation_matrix_heatmap(
        df: pd.DataFrame,
        x_features: Sequence[str],
        y_features: Sequence[str],
        *,
        cluster_rows: bool = True,
        cluster_cols: Optional[bool] = None,
        method: str = "pearson",
        cmap: str = "coolwarm",
    ):

        if cluster_cols is None:
            cluster_cols = cluster_rows

        if x_features is None:
            x_features = df.columns.to_list()
        if y_features is None:
            y_features = x_features  # retains square behaviour

        X = (
            df[x_features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        Y = (
            df[y_features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        # keep only informative columns
        X = X.loc[:, X.nunique(dropna=True) > 1]
        Y = Y.loc[:, Y.nunique(dropna=True) > 1]

        if X.empty or Y.empty:
            raise ValueError("Need at least one valid feature on each axis.")

        # ── Rectangular correlation matrix ───────────────────────────────────────
        corr = pd.DataFrame({
            col_y: X.corrwith(Y[col_y], method=method) for col_y in Y.columns
        })

        # ── Optional clustering ─────────────────────────────────────────────────
        if cluster_rows and corr.shape[0] > 1:
            sim_rows = corr.T.corr().abs().fillna(0)
            row_linkage = sch.linkage(
                squareform(1 - sim_rows.values, checks=False),
                method="average",
                optimal_ordering=True,
            )
            row_order = sch.dendrogram(row_linkage, no_plot=True)["leaves"]
            corr = corr.iloc[row_order, :]

        if cluster_cols and corr.shape[1] > 1:
            sim_cols = corr.corr().abs().fillna(0)
            col_linkage = sch.linkage(
                squareform(1 - sim_cols.values, checks=False),
                method="average",
                optimal_ordering=True,
            )
            col_order = sch.dendrogram(col_linkage, no_plot=True)["leaves"]
            corr = corr.iloc[:, col_order]

        height = max(4, corr.shape[0] * 0.35)
        width = max(4, corr.shape[1] * 0.35)
        plt.figure(figsize=(width, height))

        sns.heatmap(
            corr,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.4,
            linecolor="black",
            square=False,
            cbar_kws={"label": f"{method.capitalize()} r"},
        )

        plt.xticks(rotation=90, ha="center", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()

    plot_cross_correlation_matrix_heatmap(features_df, xcats["kpms-v2_all"], xcats["supervised"])
    return


@app.cell
def _(Optional, Tuple, pd, plt, results_df, sns):
    def plot_model_performance_bar_graph(
        df: pd.DataFrame,
        *,
        figsize: Tuple[float, float]       = (3., 6.),
        lim: Optional[Tuple[float, float]] = None,
        ylabel: str                        = "MAE",
    ):
        abs_err = (df["y_pred"] - df["y_true"]).abs()
        fold_mae = (
            abs_err.groupby([df["model"], df["fold"]])
            .mean()
            .reset_index(name="mae")
        )

        assert fold_mae.groupby("model").size().eq(10).all(), "Some model is missing folds."

        plt.figure(figsize=figsize)
        sns.boxplot(
            x="model",
            y="mae",
            data=fold_mae,
            showfliers=True,
            width=0.4,
            flierprops={"marker": ".", "color": "black", "markersize": 4},
        )
        plt.xticks(rotation=90)
        if lim:
            plt.ylim(*lim)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xlabel("Model")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    plot_model_performance_bar_graph(
        results_df[
            (results_df["y_cat"] == "age") &
            (results_df["X_cat"] == "kpms-v2_all") &
            (results_df["split"]  == "test") &
            (results_df["model"].isin(["Elastic Net", "Random Forest", "XGBoost", "Multi-Layer Perceptron"]))
        ],
        lim = (8., 22.),
        ylabel = "MAE (weeks)",
    )
    return


@app.cell
def _(Mapping, Optional, Sequence, Tuple, pd, plt, sns):
    def plot_grouped_model_performance_boxplot(
        dfs: Sequence[pd.DataFrame],
        label_map: Optional[Mapping[str, str]] = None,
        *,
        figsize: Tuple[float, float]       = (6.0, 6.0),
        lim: Optional[Tuple[float, float]] = None,
        ylabel: str                        = "MAE",
    ) -> None:
        frames = []
        for df in dfs:
            mae = (df["y_pred"] - df["y_true"]).abs()
            frame = (
                mae.groupby([df["model"], df["fold"]])
                .mean()
                .reset_index(name="mae")
            )
            frame["X_cat"] = df["X_cat"].iloc[0]
            frames.append(frame)

        fold_mae_all = pd.concat(frames, ignore_index=True)
        fold_mae_all["slice_label"] = (
            fold_mae_all["X_cat"].map(label_map) if label_map else fold_mae_all["X_cat"]
        )

        assert fold_mae_all.groupby(["model", "X_cat"]).size().eq(10).all(), "Some model is missing folds."

        plt.figure(figsize=figsize)
        sns.boxplot(
            data=fold_mae_all,
            x="model",
            y="mae",
            hue="slice_label",
            dodge=True,
            width=0.6,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 4, "color": "black"},
        )
        plt.xticks(rotation=90)
        if lim:
            plt.ylim(*lim)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xlabel("Model")
        plt.ylabel(ylabel)
        plt.legend(title=None, loc="upper right")
        plt.tight_layout()
        plt.show()

    # from plotnine import (
    #     ggplot, aes, geom_boxplot, position_dodge,
    #     scale_y_continuous, labs, theme_bw, theme,
    #     element_text, element_line, element_blank 
    # )


    # def plot_grouped_model_performance_boxplot(
    #     dfs: Sequence[pd.DataFrame],
    #     label_map: Optional[Mapping[str, str]] = None,
    #     *,
    #     figsize: Tuple[float, float]       = (6.0, 6.0),
    #     lim: Optional[Tuple[float, float]] = None,
    #     ylabel: str                        = "MAE",
    # ):
    #     """
    #     Draw a grouped-boxplot of model performance, sliced by X_cat,
    #     using plotnine (ggplot syntax).

    #     Parameters
    #     ----------
    #     dfs       : iterable of DataFrames
    #         Each DataFrame must contain columns
    #         ['y_pred', 'y_true', 'model', 'fold', 'X_cat'].
    #     label_map : dict, optional
    #         Maps raw slice labels in `X_cat` to pretty names.
    #     figsize   : (width, height) in inches for the final figure.
    #     lim       : (ymin, ymax) to constrain the y-axis; pass None for auto.
    #     ylabel    : str, y-axis label (default “MAE”).

    #     Returns
    #     -------
    #     plotnine.ggplot
    #         A ggplot object that will render automatically in Jupyter
    #         (or can be printed / saved).
    #     """
    #     # ── Aggregate MAE per (model, fold, slice) ────────────────────────────
    #     frames = []
    #     for df in dfs:
    #         mae = (df["y_pred"] - df["y_true"]).abs()
    #         frame = (
    #             mae.groupby([df["model"], df["fold"]])
    #             .mean()
    #             .reset_index(name="mae")
    #         )
    #         frame["X_cat"] = df["X_cat"].iloc[0]      # add slice identifier
    #         frames.append(frame)

    #     fold_mae_all = pd.concat(frames, ignore_index=True)
    #     fold_mae_all["slice_label"] = (
    #         fold_mae_all["X_cat"].map(label_map)
    #         if label_map else
    #         fold_mae_all["X_cat"]
    #     )

    #     # sanity-check: every (model, slice) should have 10 folds
    #     assert fold_mae_all.groupby(["model", "X_cat"]).size().eq(10).all(), \
    #         "Some model is missing folds."

    #     # ── Build the plot ────────────────────────────────────────────────────
    #     g = (
    #         ggplot(fold_mae_all,
    #                aes(x="model",
    #                    y="mae",
    #                    fill="slice_label"))
    #         + geom_boxplot(
    #             width=.6,
    #             position=position_dodge(width=.6),
    #             outlier_shape=".",
    #             outlier_size=2
    #         )
    #         + labs(x="Model", y=ylabel, fill=None)
    #         + theme_bw()
    #         + theme(
    #             figure_size=figsize,
    #             axis_text_x=element_text(angle=90, va="center"),
    #             panel_grid_major_y=element_line(linetype="--", alpha=.7),   # ✓ correct
    #             panel_grid_major_x=element_blank()                          # ✓ hide vertical gridlines
    #         )
    #     )

    #     if lim:
    #         g += scale_y_continuous(limits=lim)

    #     return g
    return (plot_grouped_model_performance_boxplot,)


@app.cell
def _(plot_grouped_model_performance_boxplot, results_df):
    def plot_ycat_kpmsv1_supunsup_results(ycat: str, ylabel: str, **kwargs):
        g = plot_grouped_model_performance_boxplot(
            [
                # results_df[
                #     (results_df["y_cat"] == ycat) &
                #     (results_df["X_cat"] == "kpms-v2_nonmeta") &
                #     (results_df["split"]  == "test") &
                #     (results_df["model"].isin(["Elastic Net", "Random Forest", "XGBoost", "Multi-Layer Perceptron"]))
                # ],
                results_df[
                    (results_df["y_cat"] == ycat) &
                    (results_df["X_cat"] == "kpms-v2_all") &
                    (results_df["split"]  == "test") &
                    (results_df["model"].isin(["Elastic Net", "Random Forest", "XGBoost", "Multi-Layer Perceptron"]))
                ],
                results_df[
                    (results_df["y_cat"] == ycat) &
                    (results_df["X_cat"] == "supervised") &
                    (results_df["split"]  == "test") &
                    (results_df["model"].isin(["Elastic Net", "Random Forest", "XGBoost", "Multi-Layer Perceptron"]))
                ],
                results_df[
                    (results_df["y_cat"] == ycat) &
                    (results_df["X_cat"] == "all") &
                    (results_df["split"]  == "test") &
                    (results_df["model"].isin(["Elastic Net", "Random Forest", "XGBoost", "Multi-Layer Perceptron"]))
                ],

            ],
            {
                "kpms-v2_all": "unsup. w/o metasyllables",
                "supervised": "supervised features",
                "all": "unsupervised + supervised features",
            },
            ylabel=ylabel,
            **kwargs,
        )
        print(g)

    plot_ycat_kpmsv1_supunsup_results("age", ylabel="MAE (weeks)", lim = (8., 30.))
    plot_ycat_kpmsv1_supunsup_results("fi", ylabel="MAE (FI)", lim = (1.0, 4))
    return


@app.cell
def _(Optional, Tuple, linregress, np, pd, plt, results_df):
    def plot_pred_vs_true(
        df: pd.DataFrame,
        x_cat: str,
        y_cat: str,
        *,
        test_only: bool = True,            # switch off to include train/val
        split_col: str = "split",
        y_true_col: str = "y_true",
        y_pred_col: str = "y_pred",
        x_cat_col: str = "X_cat",
        y_cat_col: str = "y_cat",
        lims: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Scatter-plot predicted vs. true values (optionally test-set only)
        with a 45° reference line, a line of best fit, and
        printed diagnostics.

        Parameters
        ----------
        df : DataFrame
            Your master results table (one row per sample-fold combination).
        x_cat, y_cat : str
            Filter to these categories.
        test_only : bool
            Keep only rows whose `split` column is 'test' (case-insensitive).
        *_col : str
            Override column names if yours differ.
        lims : (low, high)
            Fix axis limits; if None they’re inferred from the data.
        """

        mask = pd.Series(True, index=df.index)

        if test_only:
            mask &= df[split_col].str.lower() == "test"
        mask &= df[x_cat_col] == x_cat
        mask &= df[y_cat_col] == y_cat

        sub = df.loc[mask]

        if sub.empty:
            print("❗ No rows match the chosen filters.")
            return

        n_points = len(sub)
        print(f"Plotted {n_points} points.")

        slope, intercept, r_value, _, _ = linregress(sub[y_true_col], sub[y_pred_col])
        r2 = r_value**2
        print(f"Slope = {slope:.3f}   R² = {r2:.3f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(sub[y_true_col], sub[y_pred_col], alpha=0.6, s=20)

        if lims is None:
            lims = [
                min(sub[[y_true_col, y_pred_col]].min()),
                max(sub[[y_true_col, y_pred_col]].max()),
            ]

        plt.plot(lims, lims, ls="--", label="Perfect prediction")

        # best-fit line
        fit_x = np.array(lims)
        fit_y = slope * fit_x + intercept
        plt.plot(fit_x, fit_y, label=f"Best fit (m={slope:.2f}, R²={r2:.2f})")

        plt.xlim(lims)
        plt.ylim(lims)

        plt.title(f"Predicted vs True | X_cat: {x_cat} | y_cat: {y_cat}")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_pred_vs_true(results_df[results_df["model"] == "XGBoost"], "all", "age", lims=(0, 200))
    plot_pred_vs_true(results_df[results_df["model"] == "XGBoost"], "all", "fi", lims=(0, 14))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
