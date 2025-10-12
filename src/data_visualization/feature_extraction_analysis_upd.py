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
    do_df.head(3)
    return b6j_df, do_df


@app.cell
def _(b6j_df, do_df):
    def get_syllable_stats(df):
        cols = [c for c in df.columns if c.startswith("syllable_frequency")]
        return {c: (df[c].mean(), df[c].std()) for c in cols}

    b6j_stats = get_syllable_stats(b6j_df)
    do_stats = get_syllable_stats(do_df)
    return b6j_stats, do_stats


@app.cell
def _(b6j_stats):
    b6j_stats
    return


@app.cell
def _(b6j_stats, do_stats, plt):
    b6j_means = {c: v[0] for c, v in b6j_stats.items()}
    do_means = {c: v[0] for c, v in do_stats.items()}
    normalized_do = {c: do_means[c] / b6j_means[c] for c in b6j_means}

    x = list(range(1, len(b6j_means) + 1))
    y_do = list(normalized_do.values())

    plt.figure(figsize=(10,5))
    plt.plot(x, [1]*len(x), label="B6J (baseline)")
    plt.plot(x, y_do, label="DO / B6J", marker='o')
    plt.xticks(ticks=x[::5], labels=[i for i in x[::5]])
    plt.xlabel("Syllable Number")
    plt.ylabel("Normalized syllable usage")
    plt.title("Normalized syllable frequency: DO vs B6J")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return b6j_means, normalized_do, x, y_do


@app.cell
def _(b6j_means, normalized_do, plt):
    syllable_nums = [int(c.split("_")[-1]) for c in b6j_means.keys()]
    filtered = [(num, normalized_do[c]) for c, num in zip(b6j_means.keys(), syllable_nums) if num != 69]

    _x, _y_do = zip(*sorted(filtered))
    _x = _x[:53]
    _y_do = _y_do[:53]

    plt.figure(figsize=(10,5))
    plt.plot(_x, [1]*len(_x), label="B6J (baseline)")
    plt.plot(_x, _y_do, label="DO / B6J", marker='o')
    plt.xticks(ticks=_x[::5], labels=[i for i in _x[::5]])
    plt.yscale("log")
    plt.xlabel("Syllable Number")
    plt.ylabel("Normalized syllable usage")
    plt.title("Normalized syllable frequency: DO vs B6J")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, x, y_do):
    plt.figure(figsize=(10,5))
    plt.plot(x, [1]*len(x), label="B6J (baseline)")
    plt.plot(x, y_do, label="DO / B6J", marker='o')
    plt.yscale("log")
    plt.xticks(ticks=x[::5], labels=[i for i in x[::5]])
    plt.xlabel("Syllable Number")
    plt.ylabel("Normalized syllable usage (log scale)")
    plt.title("Normalized syllable frequency: DO vs B6J (excluding syllable 69)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(b6j_stats, do_stats):
    print(b6j_stats["syllable_frequency_70"], do_stats["syllable_frequency_70"])
    return


@app.cell
def _(b6j_stats, do_stats, np, plt):
    _syllable_nums = [int(c.split("_")[-1]) for c in b6j_stats.keys()]
    _syllable_nums = [c for c in _syllable_nums if c <= 53]
    _filtered = [(num, b6j_stats[c], do_stats[c]) for c, num in zip(b6j_stats.keys(), _syllable_nums)]
    _filtered = sorted(_filtered, key=lambda x: x[0])

    _x_labels = [f"S{i}" for i, _, _ in _filtered]
    _b6j_means = [v[0] for _, v, _ in _filtered]
    _b6j_stds = [v[1] for _, v, _ in _filtered]
    _do_means = [v[0] for _, _, v in _filtered]
    _do_stds = [v[1] for _, _, v in _filtered]

    _x = np.arange(len(_x_labels))
    _width = 0.4

    plt.figure(figsize=(12,6))
    plt.bar(_x - _width/2, _b6j_means, _width, yerr=_b6j_stds, label="B6J", capsize=3)
    plt.bar(_x + _width/2, _do_means, _width, yerr=_do_stds, label="DO", capsize=3)
    plt.yscale("log")
    plt.xticks(ticks=_x[::5], labels=[_x_labels[i] for i in range(0, len(_x_labels), 5)])
    plt.xlabel("Syllable Number")
    plt.ylabel("Syllable Frequency (log scale)")
    plt.title("Syllable Frequency Comparison: DO vs B6J (all syllables)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
