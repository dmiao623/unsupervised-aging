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
    from pathlib import Path

    mpl.style.use("default")
    return Path, json, np, os, pd, plt


@app.cell
def _(Path, json, os, pd):
    feature_matrices_dir = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices"
    feature_matrix_path = feature_matrices_dir / "2025-07-23_feature-matrix__combined_1126__2025-07-20_kpms-v4_150__2025-07-20_model-1.csv"
    xcats_path = feature_matrices_dir / "2025-07-23_xcats__combined_1126__2025-07-20_kpms-v4_150__2025-07-20_model-1.json"

    combined_df = pd.read_csv(feature_matrix_path)
    with xcats_path.open("r") as f:
        xcats = json.load(f)
    combined_df.head(3)
    return (combined_df,)


@app.cell
def _(combined_df):
    filtered_df = combined_df[["name", "mouse_id", "sex", "batch", "tester", "age", "fi", "weight", "diet", "strain", "fll", "raw_fi"] + [c for c in combined_df.columns if c.startswith("syllable_frequency")]]
    b6j_df, do_df = filtered_df.query("strain == 'B6'"), filtered_df.query("strain == 'DO'")

    q25 = filtered_df["age"].quantile(0.25)
    q75 = filtered_df["age"].quantile(0.75)

    young_df = filtered_df[filtered_df["age"] <= q25]
    old_df = filtered_df[filtered_df["age"] >= q75]
    return b6j_df, do_df, old_df, young_df


@app.cell
def _(b6j_df, do_df, old_df, young_df):
    def get_syllable_stats(df):
        cols = [c for c in df.columns if c.startswith("syllable_frequency")]
        return {c: (df[c].mean(), df[c].std()) for c in cols}

    b6j_stats = get_syllable_stats(b6j_df)
    do_stats = get_syllable_stats(do_df)
    young_stats = get_syllable_stats(young_df)
    old_stats = get_syllable_stats(old_df)
    return b6j_stats, do_stats, old_stats, young_stats


@app.cell
def _(b6j_stats, do_stats, np, plt):
    def plot_normalized_syllable_usage(baseline_means, other_means, baseline_label="B6J", other_label="DO", figsize=(10,5), xtick_step=5, show=True, log_scale=False):
        keys = [k for k in baseline_means if k in other_means]
        b = np.array([baseline_means[k] for k in keys], dtype=float)
        o = np.array([other_means[k] for k in keys], dtype=float)
        norm = np.divide(o, b, out=np.full_like(o, np.nan), where=b!=0)
        x = np.arange(1, len(keys)+1)
        fig, ax = plt.subplots(figsize=figsize)
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
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        # return fig, ax, dict(zip(keys, norm.tolist())), keys

    b6j_means = {c: v[0] for c, v in b6j_stats.items()}
    do_means = {c: v[0] for c, v in do_stats.items()}
    plot_normalized_syllable_usage(b6j_means, do_means, show=True, log_scale=True)
    return (plot_normalized_syllable_usage,)


@app.cell
def _(old_stats, plot_normalized_syllable_usage, young_stats):
    young_means = {c: v[0] for c, v in young_stats.items()}
    old_means = {c: v[0] for c, v in old_stats.items()}
    plot_normalized_syllable_usage(young_means, old_means, show=True, baseline_label="Young", other_label="Old", log_scale=True)
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
