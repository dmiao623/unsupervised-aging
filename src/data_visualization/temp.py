import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import pandas as pd

    from pathlib import Path
    return Path, os, pd


@app.cell
def _(Path, os, pd):
    df_path = Path(os.environ["UNSUPERVISED_AGING"]) / "data/feature_matrices/2025-07-23_feature-matrix__nature-aging_634__2025-07-03_kpms-v2__2025-07-07_model-2.csv"
    df = pd.read_csv(df_path)
    return (df,)


@app.cell
def _(df):
    df.loc[df['age'].idxmax()]
    return


@app.cell
def _(df):
    df.loc[df['age'].idxmin()]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
