import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import seaborn as sns

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from pathlib import Path
    from typing import Dict, Tuple

    mpl.style.use("default")
    return Dict, Line2D, Patch, Path, Tuple, json, os, pd, plt


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
def _(data):
    _df1 = data["nature-aging_634"]["features"]
    _df2 = data["geroscience_492"]["features"]

    (
        len(set(_df1["mouse_id"])),
        len(set(_df2["mouse_id"]))
    )
    (
        (min(_df1["age"]), max(_df1["age"])),
        (min(_df2["age"]), max(_df2["age"]))
    )
    return


@app.cell
def _(Dict, Tuple, data, pd, plt):
    def _plot_diet_distribution(
        *,
        diet_colors: Dict[str, str] = {
            "AL": "gray", "1D": "lightblue", "2D": "blue",
            "20": "orange", "40": "red"
        },
        figsize: Tuple[int, int] = (3, 6),
        fontsize: int = 10,
    ):
        df_tmp = data["nature-aging_634"]["features"].copy()
        df_tmp["diet"] = "AL"

        combined = pd.DataFrame({
            "B6J": df_tmp["diet"].value_counts(),
            "DO": data["geroscience_492"]["features"]["diet"].value_counts()
        }).fillna(0)

        combined = combined.loc[combined.index.intersection(diet_colors.keys())]
        combined = combined.loc[sorted(combined.index, key=lambda x: list(diet_colors).index(x))]

        plt.figure(figsize=figsize, dpi=300)
        ax = combined.T.plot(
            kind="bar",
            stacked=True,
            color=[diet_colors[diet] for diet in combined.index],
            ax=plt.gca()
        )
        ax.legend_.remove()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylabel("Count", fontsize=fontsize)
        ax.set_xlabel("Strain", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

        plt.tight_layout()
        plt.show()
    _plot_diet_distribution(fontsize=20)
    return


@app.cell
def _(Line2D, Patch, data, pd, plt):
    def _plot_age_fi_scatter(
        *,
        diet_colors={"AL": "gray", "1D": "lightblue", "2D": "blue", "20": "orange", "40": "red"},
        figsize=(7, 5),
        fontsize=10,
    ):
        df = data["combined_1126"]["features"].copy()
        strain_markers = {"B6": "o", "DO": "^"}

        plt.figure(figsize=figsize, dpi=300)
        labeled_diets = set()

        for strain in df.strain.unique():
            for diet in df.diet.unique():
                subset = df[(df.strain == strain) & (df.diet == diet)]
                label = diet if diet not in labeled_diets else None
                if label:
                    labeled_diets.add(diet)
                plt.scatter(
                    subset.age,
                    subset.fi,
                    label=label,
                    c=diet_colors[diet],
                    marker=strain_markers[strain],
                    alpha=0.6,
                    edgecolors="w",
                    s=60,
                )

        df["age_bin"] = pd.cut(df.age, bins=[0, 50, 100, 150, 200])
        summary = (
            df.groupby(["strain", "age_bin"]).fi.agg(["mean", "std"]).reset_index()
        )
        summary["age_mid"] = summary.age_bin.map(lambda x: x.mid)

        for strain in summary.strain.unique():
            sub = summary[summary.strain == strain].sort_values("age_mid")
            plt.errorbar(
                sub.age_mid,
                sub["mean"],
                yerr=sub["std"].fillna(0),
                fmt="-o" if strain == "B6" else "-^",
                color="black",
                linewidth=2,
                capsize=3,
            )

        diet_handles = [Patch(facecolor=c, label=d) for d, c in diet_colors.items()]
        strain_handles = [
            Line2D([0], [0], marker=m, color="w", label=s, markerfacecolor="black", markersize=10)
            for s, m in strain_markers.items()
        ]

        first = plt.legend(handles=diet_handles, title="Diet", bbox_to_anchor=(1.05, 0.3), loc="center left", fontsize=fontsize, title_fontsize=fontsize)
        plt.gca().add_artist(first)
        plt.legend(handles=strain_handles, title="Strain", bbox_to_anchor=(1.05, 0.9), loc="upper left", fontsize=fontsize, title_fontsize=fontsize)

        plt.xlabel("Age (weeks)", fontsize=fontsize)
        plt.ylabel("Score (CFI)", fontsize=fontsize)
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", labelsize=fontsize)
        plt.tight_layout()
        plt.show()


    _plot_age_fi_scatter(fontsize=16)
    return


@app.cell
def _(Dict, Line2D, Patch, Tuple, data, pd, plt):
    def plot_combined_diet_age_fi(
        *,
        diet_colors: Dict[str, str] = {
            "AL": "gray", "1D": "lightblue", "2D": "blue",
            "20": "orange", "40": "red"
        },
        figsize: Tuple[int, int] = (10, 6),
        wspace: float = 0.3, 
    ):
        # Prepare diet distribution data
        df_tmp = data["nature-aging_634"]["features"].copy()
        df_tmp["diet"] = "AL"

        combined = pd.DataFrame({
            "B6J": df_tmp["diet"].value_counts(),
            "DO": data["geroscience_492"]["features"]["diet"].value_counts()
        }).fillna(0)

        combined = combined.loc[combined.index.intersection(diet_colors.keys())]
        combined = combined.loc[sorted(combined.index, key=lambda x: list(diet_colors).index(x))]

        # Prepare scatter data
        df = data["combined_1126"]["features"].copy()
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

        diet_legend = [Patch(facecolor=color, label=diet) for diet, color in diet_colors.items()]
        strain_legend = [Line2D([0], [0], marker=marker, color='w',
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
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
