import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


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

    from matplotlib.colors import to_rgba
    from pathlib import Path
    from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Sequence

    mpl.style.use("default")
    return Path, json, np, os, pd, plt, to_rgba


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
def _(data, plt):
    def _plot_manual_vs_adjusted_fis():
        df = data["combined_1126"]["features"]
        fig, ax = plt.subplots(dpi=300)

        testers = df["tester"].unique()
        cmap = plt.get_cmap("viridis", len(testers))

        for i, tester in enumerate(testers):
            d = df[df["tester"] == tester]
            ax.scatter(d["raw_fi"], d["fi"], color=cmap(i), label=tester, alpha=0.7)

        ax.set_xlabel("Raw FI")
        ax.set_ylabel("Adjusted FI")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show()

    _plot_manual_vs_adjusted_fis()

    return


@app.cell
def _(data, np, plt, to_rgba):
    def plot_feature_correlations(
        dataset_id,
        syllable_group_prefixes,
        xcat,
        cmap="coolwarm",
        n_features=None,
        per_group=False,
        cell_size=0.4,
        bar_height_factor=0.4,  # new argument (color-bar height = cell_size * bar_height_factor)
        label_fontsize=10,
        legend_fontsize=10,
        legend_space=0.08,
        grid=False,
        show_legend=True
    ):
        prefix_to_color = {v[1]: k for k, v in syllable_group_prefixes.items()}
        color_labels = {k: v[0] for k, v in syllable_group_prefixes.items()}

        df = data[dataset_id]["features"]
        feats_all = list(data[dataset_id]["xcats"][xcat])
        corr_fi = df[feats_all].corrwith(df["fi"])
        corr_age = df[feats_all].corrwith(df["age"])
        abs_score = np.maximum(np.abs(corr_fi), np.abs(corr_age))

        if n_features is None:
            idx = range(len(feats_all))
        elif per_group and isinstance(n_features, (tuple, list)) and len(n_features) == 3:
            grp_idx = {g: [] for g in prefix_to_color}
            for i, f in enumerate(feats_all):
                key = next((p for p in prefix_to_color if f.startswith(p)), None)
                if key:
                    grp_idx[key].append(i)
            sel = []
            for key, n in zip(prefix_to_color, n_features):
                if n is None:
                    sel += grp_idx[key]
                else:
                    s = abs_score.iloc[grp_idx[key]].values
                    top = np.argsort(-s)[:n]
                    sel += [grp_idx[key][t] for t in top]
            idx = sorted(sel)
        else:
            idx = sorted(np.argsort(-abs_score.values)[:n_features])

        feats = [feats_all[i] for i in idx]
        corr = np.vstack([corr_fi.iloc[idx].values, corr_age.iloc[idx].values])
        bar = [
            prefix_to_color.get(next((p for p in prefix_to_color if f.startswith(p)), None), "lightgrey")
            for f in feats
        ]
        bar_rgba = np.array([[to_rgba(c) for c in bar]])

        width = len(feats) * cell_size + 1
        height_heat = 2 * cell_size
        height_bar = cell_size * bar_height_factor
        fig, (ax0, ax1) = plt.subplots(
            2, 1,
            figsize=(width, height_heat + height_bar),
            gridspec_kw={"height_ratios": [height_heat, height_bar]},
            sharex=True
        )

        ax0.imshow(corr, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        ax0.set_yticks([0, 1])
        ax0.set_yticklabels(["FI", "Age"], fontsize=label_fontsize)
        ax0.set_xticks([])

        if grid:
            ax0.set_xticks(np.arange(corr.shape[1] + 1) - 0.5, minor=True)
            ax0.set_yticks(np.arange(corr.shape[0] + 1) - 0.5, minor=True)
            ax0.grid(which="minor", linewidth=0.3)
            ax0.tick_params(which="minor", bottom=False, left=False)

        ax1.imshow(bar_rgba, aspect="auto")
        ax1.set_xticks([])
        ax1.set_yticks([])

        if show_legend:
            handles = [plt.Line2D([0], [0], marker="s", color=c, linestyle="", markersize=10)
                       for c in syllable_group_prefixes]
            labels = [color_labels[c] for c in syllable_group_prefixes]
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 1 - legend_space / 2),
                       ncol=3, frameon=False, fontsize=legend_fontsize)

        plt.tight_layout(rect=[0, 0, 1, 1 - (legend_space if show_legend else 0)])
        plt.show()


    plot_feature_correlations(
        "combined_1126",
        {
            "#014421": ("Embedding", "latent_embedding_"),
            "#9bd4a3": ("Syllable", "syllable_frequency_"),
            "#bfb239": ("Transition", "transition_matrix_")
        },
        "unsupervised",
        cell_size=0.6,
        n_features=(20, 20, 20),
        per_group=True,
        label_fontsize=40,
        legend_fontsize=0,
        legend_space=-0.1,
        show_legend=False,
        grid=True,
        bar_height_factor=0.6
    )
    return (plot_feature_correlations,)


@app.cell
def _(plot_feature_correlations):
    plot_feature_correlations(
        "combined_1126",
        {
            "#885791": ("Supervised", ""),
        },
        "supervised",
        cell_size=0.6,
        n_features=60,
        label_fontsize=40,
        legend_fontsize=0,
        legend_space=-0.1,
        show_legend=False,
        grid=True,
        bar_height_factor=0.6
    )
    return


@app.cell
def _(data):
    _dataset_id = "geroscience_492"
    xcat__ = data[_dataset_id]["xcats"]
    _df1 = data[_dataset_id]["features"]
    test1_age = _df1[xcat__["unsupervised"]].corrwith(_df1["age"])
    test1_fi  = _df1[xcat__["unsupervised"]].corrwith(_df1["fi"])
    test2_age = _df1[xcat__["supervised"]].corrwith(_df1["age"])
    test2_fi  = _df1[xcat__["supervised"]].corrwith(_df1["fi"])
    return test1_age, test1_fi, test2_age, test2_fi


@app.cell
def _(plt, test1_age, test1_fi, test2_age, test2_fi):
    plt.scatter(test1_age, test1_fi, color="tab:blue", label="Unsupervised", s=60)
    plt.scatter(test2_age, test2_fi, color="tab:orange", label="Supervised", s=60)
    plt.xlabel("Correlation with Age")
    plt.ylabel("Correlation with FI")
    plt.axhline(0, linewidth=0.8, linestyle="--")
    plt.axvline(0, linewidth=0.8, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(data, plt):
    def plot_feature_correlation_scatter(
        dataset_id,
        color_map,
        show_legend=True,
        xscale="linear",
        yscale="linear",
        xlim=None,
        ylim=None,
        point_size=40,
        figsize=None,
        title=None,
        text_size=12,
    ):
        if figsize is not None:
            plt.figure(figsize=figsize)

        xcat = data[dataset_id]["xcats"]
        df = data[dataset_id]["features"]

        for color, (label, prefix) in color_map.items():
            feats = xcat["supervised"] if prefix == "" else [f for f in xcat["unsupervised"] if f.startswith(prefix)]
            if not feats:
                continue
            x = df[feats].corrwith(df["age"])
            y = df[feats].corrwith(df["fi"])
            plt.scatter(x, y, s=point_size, color=color, label=label)

        plt.xlabel("Correlation with Age", fontsize=text_size)
        plt.ylabel("Correlation with FI", fontsize=text_size)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xticks(fontsize=text_size)
        plt.yticks(fontsize=text_size)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.axhline(0, linewidth=0.8, linestyle="--")
        plt.axvline(0, linewidth=0.8, linestyle="--")

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_legend:
            plt.legend(fontsize=text_size * 0.9)
        if title is not None:
            plt.title(title, fontsize=text_size * 1.1)

        plt.tight_layout()
        plt.show()


    color_map = {
        "#014421": ("Embedding", "latent_embedding_"),
        "#9bd4a3": ("Syllable", "syllable_frequency_"),
        "#bfb239": ("Transition", "transition_matrix_"),
        "#885791": ("Supervised", ""),
    }

    plot_feature_correlation_scatter(
        "combined_1126",
        color_map,
        show_legend=False,
        xscale="linear",
        yscale="linear",
        xlim=(-0.75, 0.75),
        ylim=(-0.75,0.75),
        figsize=(6,6),
        title="",
        text_size=18
    )

    return (color_map,)


@app.cell
def _(color_map, plt):
    import matplotlib.patches as mpatches


    handles = [mpatches.Patch(color=c, label=n) for c, (n, _) in color_map.items()]
    fig, ax = plt.subplots(figsize=(6, 0.4))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
    plt.tight_layout()
    plt.show()
    return (mpatches,)


@app.cell
def _(np, plt):

    def plot_coolwarm_scale(figsize=(6, 1), vmin=-1, vmax=1, n_ticks=5):
        fig, ax = plt.subplots(figsize=figsize)
        gradient = np.linspace(vmin, vmax, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap='coolwarm', extent=[vmin, vmax, 0, 0.5])
        ax.set_yticks([])
        ax.set_xticks(np.linspace(vmin, vmax, n_ticks))
        ax.set_xlim(vmin, vmax)
        plt.title("Pearson correlation")
        plt.tight_layout()
        plt.show()

    plot_coolwarm_scale(figsize=(4, 0.5))
    return


@app.cell
def _(data, plt):
    def plot_feature_strain_correlation_scatter(
        dataset_id,
        color_map,
        y_cat="fi",
        show_legend=True,
        xscale="linear",
        yscale="linear",
        xlim=None,
        ylim=None,
        point_size=40,
        figsize=None,
        title=None,
        font_size=12          # new argument
    ):
        if y_cat not in ("fi", "age"):
            raise ValueError("y_cat must be 'fi' or 'age'")
        if figsize is not None:
            plt.figure(figsize=figsize)

        xcat = data[dataset_id]["xcats"]
        df = data[dataset_id]["features"]
        df_B6 = df[df["strain"] == "B6"]
        df_DO = df[df["strain"] == "DO"]

        for color, (label, prefix) in color_map.items():
            feats = xcat["supervised"] if prefix == "" else [f for f in xcat["unsupervised"] if f.startswith(prefix)]
            if not feats:
                continue
            x = df_B6[feats].corrwith(df_B6[y_cat]).abs()
            y = df_DO[feats].corrwith(df_DO[y_cat]).abs()
            plt.scatter(x, y, s=point_size, color=color, label=label)

        plt.xlabel(r"$|r|_{\text{Age}, \text{feature}}(\text{B6J})$", fontsize=font_size)
        plt.ylabel(r"$|r|_{\text{Age}, \text{feature}}(\text{DO})$", fontsize=font_size)
        plt.xscale(xscale)
        plt.yscale(yscale)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()
        lim_min, lim_max = min(xmin, ymin), max(xmax, ymax)
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "--", linewidth=0.8)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.axhline(0, linewidth=0.8, linestyle="--")
        plt.axvline(0, linewidth=0.8, linestyle="--")

        ax.tick_params(labelsize=font_size)
        if show_legend:
            plt.legend(fontsize=font_size)
        if title:
            plt.title(title, fontsize=font_size + 2)
        plt.tight_layout()
        plt.show()


    _color_map = {
        "#014421": ("Embedding", "latent_embedding_"),
        "#9bd4a3": ("Syllable", "syllable_frequency_"),
        "#bfb239": ("Transition", "transition_matrix_"),
        # "#885791": ("Supervised", ""),
    }

    plot_feature_strain_correlation_scatter(
        "combined_1126",
        _color_map,
        y_cat="age",
        show_legend=False,
        xscale="linear",
        yscale="linear",
        xlim=(0, 0.75),
        ylim=(0, 0.75),
        figsize=(6, 6),
        title="",
        font_size=20,
    )


    return


@app.cell
def _(data, np, plt, to_rgba):
    def plot_unsup_sup_corr_matrix(
        dataset_id,
        unsup_prefixes,
        sup_prefixes,
        unsup_xcat,
        sup_xcat,
        cmap="coolwarm",
        n_unsup=None,
        per_group=False,
        n_sup=None,
        cell_size=0.4,
        bar_size=0.4,          # thickness of the colour bars
        gap=0.02,
        grid=False,
        show_legend=True,
        legend_fontsize=10,
        legend_space=0.08,
        title=None,
    ):
        upx = {v[1]: k for k, v in unsup_prefixes.items()}
        spx = {v[1]: k for k, v in sup_prefixes.items()}

        df = data[dataset_id]["features"]
        u_all = list(data[dataset_id]["xcats"][unsup_xcat])
        s_all = list(data[dataset_id]["xcats"][sup_xcat])

        if per_group and isinstance(n_unsup, (list, tuple)) and len(n_unsup) == len(upx):
            corr_fi = df[u_all].corrwith(df["fi"])
            corr_age = df[u_all].corrwith(df["age"])
            abs_score = np.maximum(np.abs(corr_fi), np.abs(corr_age))
            grp_idx = {g: [] for g in upx}
            for i, f in enumerate(u_all):
                key = next((p for p in upx if f.startswith(p)), None)
                if key:
                    grp_idx[key].append(i)
            sel = []
            for key, n in zip(upx, n_unsup):
                s = abs_score.iloc[grp_idx[key]].values
                top = np.argsort(-s)[:n] if n is not None else np.arange(len(grp_idx[key]))
                sel += [grp_idx[key][t] for t in top]
            u_idx = sorted(sel)
        else:
            u_idx = range(len(u_all)) if n_unsup is None else np.argsort(-np.abs(df[u_all].corrwith(df["fi"]).values))[:n_unsup]

        u = [u_all[i] for i in u_idx]
        s = s_all if n_sup is None else s_all[:n_sup]

        corr = np.array([df[u].corrwith(df[y]).values for y in s])
        u_bar = [upx.get(next((p for p in upx if f.startswith(p)), None), "lightgrey") for f in u]
        s_bar = [spx.get(next((p for p in spx if f.startswith(p)), None), "lightgrey") for f in s]

        if isinstance(bar_size, (list, tuple)) and len(bar_size) == 2:
            row_bar, col_bar = bar_size
        else:
            row_bar = col_bar = bar_size

        height = len(s) * cell_size
        width = len(u) * cell_size
        fig = plt.figure(figsize=(width + 1, height + 1))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[cell_size * row_bar, width],
            height_ratios=[height, cell_size * col_bar],
            wspace=gap, hspace=gap
        )

        ax_row = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_col = fig.add_subplot(gs[1, 1])

        ax_row.imshow(np.array([[to_rgba(c)] for c in s_bar]), aspect="auto")
        ax_row.set_xticks([]); ax_row.set_yticks([])
        ax_col.imshow(np.array([[to_rgba(c) for c in u_bar]]), aspect="auto")
        ax_col.set_xticks([]); ax_col.set_yticks([])
        ax_heat.imshow(corr, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        ax_heat.set_xticks([]); ax_heat.set_yticks([])

        if grid:
            ax_heat.set_xticks(np.arange(corr.shape[1] + 1) - 0.5, minor=True)
            ax_heat.set_yticks(np.arange(corr.shape[0] + 1) - 0.5, minor=True)
            ax_heat.grid(which="minor", color="gray", linewidth=0.3)
            ax_heat.tick_params(which="minor", bottom=False, left=False)

        if show_legend:
            handles = [plt.Line2D([0], [0], marker="s", color=c, linestyle="", markersize=10)
                       for c in list(unsup_prefixes) + list(sup_prefixes)]
            labels = [v[0] for v in unsup_prefixes.values()] + [v[0] for v in sup_prefixes.values()]
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 1 - legend_space / 2),
                       ncol=3, frameon=False, fontsize=legend_fontsize)

        plt.tight_layout(rect=[0, 0, 1, 1 - (legend_space if show_legend else 0)])
        if title is not None:
            plt.suptitle(title, y=0.93, fontsize=40)
        plt.show()

    plot_unsup_sup_corr_matrix(
        "combined_1126",
        {
            "#014421": ("Embedding", "latent_embedding_"),
            "#9bd4a3": ("Syllable", "syllable_frequency_"),
            "#bfb239": ("Transition", "transition_matrix_")
        },
        {
            "#885791": ("Supervised", "")
        },
        "unsupervised",
        "supervised",
        cell_size=0.5,
        n_unsup=(20, 20, 20),
        per_group=True,
        n_sup=60,
        bar_size=1.0,      # row and column bars 0.8 × cell_size thick
        gap=0.015,
        grid=True,
        show_legend=False,
        title="Pearson Correlation between Supervised and Unsupervised Features"
    )
    return


@app.cell
def _(mpatches, plt):
    def _tmp():
        color_map = {
        "#014421": ("Embedding", "latent_embedding_"),
            "#9bd4a3": ("Syllable", "syllable_frequency_"),
            "#bfb239": ("Transition", "transition_matrix_"),
            "#885791": ("Supervised", ""),
        }

        handles = [mpatches.Patch(color=c, label=n) for c, (n, _) in color_map.items()]
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.axis("off")
        ax.legend(handles=handles, loc="center", ncol=2, frameon=False)
        plt.tight_layout()
        plt.show()
    _tmp()
    return


@app.cell
def _(data, np, plt):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    def plot_clustered_unsup_corr(
        dataset_id,
        unsup_prefixes,
        unsup_xcat,
        cmap="coolwarm",
        n_unsup=None,
        per_group=False,
        cell_size=0.4,
        gap=0.02,
        grid=False,
        title=None,
        linkage_method="ward",
        cluster_within_prefix=True
    ):
        upx = {v[1]: k for k, v in unsup_prefixes.items()}
        df = data[dataset_id]["features"]
        u_all = list(data[dataset_id]["xcats"][unsup_xcat])

        if per_group and isinstance(n_unsup, (list, tuple)) and len(n_unsup) == len(upx):
            corr_fi = df[u_all].corrwith(df["fi"])
            corr_age = df[u_all].corrwith(df["age"])
            abs_score = np.maximum(np.abs(corr_fi), np.abs(corr_age))
            grp_idx = {g: [] for g in upx}
            for i, f in enumerate(u_all):
                k = next((p for p in upx if f.startswith(p)), None)
                if k:
                    grp_idx[k].append(i)
            sel = []
            for k, n in zip(upx, n_unsup):
                s = abs_score.iloc[grp_idx[k]].values
                top = np.argsort(-s)[:n] if n is not None else np.arange(len(grp_idx[k]))
                sel.extend(grp_idx[k][t] for t in top)
            u_idx = sorted(sel)
        else:
            corr_fi = df[u_all].corrwith(df["fi"]).values
            u_idx = range(len(u_all)) if n_unsup is None else np.argsort(-np.abs(corr_fi))[:n_unsup]

        feats = [u_all[i] for i in u_idx]
        C = df[feats].corr().values

        if len(feats) > 1:
            if cluster_within_prefix and per_group:
                order = []
                for p in upx:
                    idx = [i for i, f in enumerate(feats) if f.startswith(p)]
                    if len(idx) <= 1:
                        order += idx
                    else:
                        Z = linkage(squareform(1 - C[np.ix_(idx, idx)]), method=linkage_method, optimal_ordering=True)
                        order += [idx[i] for i in leaves_list(Z)]
            else:
                Z = linkage(squareform(1 - C), method=linkage_method, optimal_ordering=True)
                order = leaves_list(Z)
            C = C[np.ix_(order, order)]

        s = len(feats) * cell_size
        fig, ax = plt.subplots(figsize=(s, s))
        ax.imshow(C, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if grid:
            ax.set_xticks(np.arange(C.shape[1] + 1) - .5, minor=True)
            ax.set_yticks(np.arange(C.shape[0] + 1) - .5, minor=True)
            ax.grid(which="minor", linewidth=0.3)
            ax.tick_params(which="minor", bottom=False, left=False)
        if title:
            plt.title(title, y=1.02, fontsize=40)
        plt.tight_layout(pad=gap)
        plt.show()
    
    plot_clustered_unsup_corr(
        "combined_1126",
        {
            "#014421": ("Embedding", "latent_embedding_"),
            "#9bd4a3": ("Syllable", "syllable_frequency_"),
            "#bfb239": ("Transition", "transition_matrix_")
        },
        "unsupervised",
        cell_size=0.5,
        n_unsup=(20, 20, 20),
        per_group=True,
        gap=0.015,
        grid=True,
        title="Clustered Correlation – Unsupervised Features"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
