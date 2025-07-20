import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import pandas as pd

    from pathlib import Path
    from plotnine import ggplot, aes, element_text, geom_col, facet_wrap, labs, theme, theme_bw, position_dodge
    from sklearn.model_selection import StratifiedShuffleSplit
    return (
        Path,
        StratifiedShuffleSplit,
        aes,
        element_text,
        facet_wrap,
        geom_col,
        ggplot,
        json,
        labs,
        os,
        pd,
        position_dodge,
        theme,
        theme_bw,
    )


@app.cell
def _(Path, os):
    dataset_dir      = Path(os.environ["UNSUPERVISED_AGING"] + "/data/datasets/geroscience_492/")
    export_dir       = dataset_dir / "kpms_training_set_200/"
    categorical_vars = ["sex", "diet"]
    strat_vars_bins  = {"age": 5}
    n_samples        = 200
    seed             = 623
    return (
        categorical_vars,
        dataset_dir,
        export_dir,
        n_samples,
        seed,
        strat_vars_bins,
    )


@app.cell
def _(
    StratifiedShuffleSplit,
    categorical_vars,
    dataset_dir,
    n_samples,
    pd,
    seed,
    strat_vars_bins,
):
    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    df = metadata_df.copy()

    for _col, _n_bins in strat_vars_bins.items():
        df[f"__bin_{_col}"] = pd.cut(
            df[_col], bins=_n_bins, labels=False, include_lowest=True
        )
    for _cat in categorical_vars:
        df[f"__bin_{_cat}"] = df[_cat].astype(str)
    strata = (
        df[[f"__bin_{c}" for c in strat_vars_bins]]
        .astype(str)
        .agg("-".join, axis=1)
    )
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples, random_state=seed
    )
    train_idx, _ = next(sss.split(df, strata))
    result = df.index[train_idx].tolist()

    assert(len(result) == n_samples)
    df
    return df, metadata_df, result


@app.cell
def _(
    aes,
    categorical_vars,
    df,
    element_text,
    facet_wrap,
    geom_col,
    ggplot,
    labs,
    pd,
    position_dodge,
    result,
    strat_vars_bins,
    theme,
    theme_bw,
):
    plot_frames = []
    all_vars = list(strat_vars_bins.keys()) + list(categorical_vars)
    for col in all_vars:
        pop = df[f"__bin_{col}"].value_counts(sort=False).rename_axis("bin").reset_index(name="count")
        pop["variable"] = col
        pop["set"] = "population"

        samp = df.loc[result, f"__bin_{col}"].value_counts(sort=False).rename_axis("bin").reset_index(name="count")
        samp["variable"] = col
        samp["set"] = "sample"

        plot_frames.extend([pop, samp])

    joint_cols = [f"__bin_{v}" for v in all_vars]
    df["__joint"] = df[joint_cols].astype(str).agg("-".join, axis=1)

    pop_joint = df["__joint"].value_counts(sort=False).rename_axis("bin").reset_index(name="count")
    pop_joint["variable"] = "joint"
    pop_joint["set"] = "population"

    samp_joint = df.loc[result, "__joint"].value_counts(sort=False).rename_axis("bin").reset_index(name="count")
    samp_joint["variable"] = "joint"
    samp_joint["set"] = "sample"

    plot_frames.extend([pop_joint, samp_joint])
    plot_df = pd.concat(plot_frames, ignore_index=True)
    plot_df["bin"] = plot_df["bin"].astype(str)

    (
        ggplot(plot_df, aes("bin", "count", fill="set"))
        + geom_col(position=position_dodge(width=0.8))
        + facet_wrap("~variable", scales="free_x")
        + labs(
            x="Bin / Joint-stratum ID",
            y="Count",
            fill="",
            title="Population vs. Sample counts"
        )
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=90, ha='right'))
    )

    return


@app.cell
def _(
    Path,
    categorical_vars,
    dataset_dir,
    export_dir,
    json,
    metadata_df,
    n_samples,
    result,
    seed,
    strat_vars_bins,
):
    videos_dir = dataset_dir / "videos"
    poses_csv_dir = dataset_dir / "poses_csv"

    dest_videos_dir = export_dir / "videos"
    dest_poses_csv_dir = export_dir / "poses_csv"

    export_dir.mkdir(exist_ok=True)
    dest_videos_dir.mkdir(exist_ok=True)
    dest_poses_csv_dir.mkdir(exist_ok=True)

    with (export_dir / "sample_params.json").open("w") as f:
        json.dump({
            "dataset_dir": str(dataset_dir),
            "export_dir": str(export_dir),
            "categorical_vars": categorical_vars,
            "strat_vars_bins": strat_vars_bins,
            "n_samples": n_samples,
            "seed": seed,
        }, f, indent=2)

    def _copy_symlink(src: Path, dst: Path, *, allow_regular: bool = False):
        if src.is_symlink():
            target = src.resolve(strict=True)
        elif allow_regular and src.is_file():
            target = src
        else:
            raise ValueError(f"{src} is not a symlink")
        if not target.exists():
            raise FileNotFoundError(f"Target {target} does not exist")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(target)

    for _name in metadata_df.loc[result, "name"]:
        _copy_symlink(
            videos_dir / f"{_name}.mp4",
            dest_videos_dir / f"{_name}.mp4",
        )
        _copy_symlink(
            poses_csv_dir / f"{_name}.csv",
            dest_poses_csv_dir / f"{_name}.csv",
            allow_regular=True,
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
