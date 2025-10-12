import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import numpy as np
    import os
    import pandas as pd

    from pathlib import Path
    from sklearn.model_selection import StratifiedShuffleSplit
    return Path, StratifiedShuffleSplit, json, np, os, pd


@app.cell
def _(Path, os):
    dataset_dir      = Path(os.environ["UNSUPERVISED_AGING"] + "/data/datasets/combined_1126/")
    export_name_base = "kpms_training_set_150_"
    categorical_vars = ["sex", "diet"]
    strat_vars_bins  = {"age": 5}
    n_samples        = 150
    base_seed        = 623
    n_splits         = 6
    return (
        base_seed,
        categorical_vars,
        dataset_dir,
        export_name_base,
        n_samples,
        n_splits,
        strat_vars_bins,
    )


@app.cell
def _(dataset_dir, pd):
    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    df = metadata_df.copy()
    df
    return df, metadata_df


@app.cell
def _(Path, StratifiedShuffleSplit, categorical_vars, np, pd, strat_vars_bins):
    def make_strata(subdf):
        tmp = subdf.copy()
        for col, n_bins in strat_vars_bins.items():
            tmp[f"__bin_{col}"] = pd.cut(tmp[col], bins=n_bins, labels=False, include_lowest=True)
        for cat in categorical_vars:
            tmp[f"__bin_{cat}"] = tmp[cat].astype(str)
        cols = [f"__bin_{c}" for c in strat_vars_bins] + [f"__bin_{c}" for c in categorical_vars]
        return tmp[cols].astype(str).agg("-".join, axis=1)

    def stratified_sample(subdf, k, seed, min_count=2):
        if k <= 0 or subdf.empty:
            return []
        rng = np.random.RandomState(seed)
        strata = make_strata(subdf)
        counts = strata.value_counts()
        rare_mask = strata.map(lambda x: counts[x] < min_count)
        rare_idx = subdf[rare_mask].index.tolist()
        take_rare = rare_idx[:min(k, len(rare_idx))]
        remaining_df = subdf.drop(index=take_rare)
        remaining_k = k - len(take_rare)
        if remaining_k <= 0:
            return take_rare[:k]
        if remaining_df.empty:
            return take_rare
        strata_rem = make_strata(remaining_df)
        rem_counts = strata_rem.value_counts()
        ok_mask = strata_rem.map(lambda x: rem_counts[x] >= min_count)
        eligible = remaining_df[ok_mask]
        if len(eligible) >= remaining_k:
            try:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=remaining_k, random_state=seed)
                idx, _ = next(sss.split(eligible, make_strata(eligible)))
                pick = eligible.index[idx].tolist()
                out = take_rare + pick
                return out
            except ValueError:
                pass
        pool = remaining_df.index.difference(take_rare)
        need = remaining_k
        if len(pool) <= need:
            return take_rare + list(pool)
        fill = rng.choice(pool, size=need, replace=False).tolist()
        return take_rare + fill


    def copy_symlink(src: Path, dst: Path, *, allow_regular: bool = False):
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
    return copy_symlink, stratified_sample


@app.cell
def _(
    base_seed,
    categorical_vars,
    copy_symlink,
    dataset_dir,
    df,
    export_name_base,
    json,
    metadata_df,
    n_samples,
    n_splits,
    strat_vars_bins,
    stratified_sample,
):
    videos_dir = dataset_dir / "videos"
    poses_csv_dir = dataset_dir / "poses_csv"

    df_do = df[df["strain"] == "DO"]
    df_b6 = df[df["strain"] == "B6"]

    for s in range(1, n_splits + 1):
        seed = base_seed + s
        n_do = len(df_do)
        n_b6 = len(df_b6)
        half = n_samples // 2
        rem = n_samples - 2 * half
        take_do = min(half + (1 if rem == 1 and n_do >= n_b6 else 0), n_do)
        take_b6 = min(n_samples - take_do, n_b6)
        if take_do + take_b6 < n_samples:
            raise ValueError("Not enough rows to meet n_samples across DO and B6.")
        idx_do = stratified_sample(df_do, take_do, seed)
        idx_b6 = stratified_sample(df_b6, take_b6, seed + 100000)
        result = idx_do + idx_b6
        if len(result) != n_samples:
            raise AssertionError("Sample size mismatch.")
        export_dir = dataset_dir / f"{export_name_base}{s}"
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
                "split": s,
                "per_strain_counts": {"DO": len(idx_do), "B6": len(idx_b6)}
            }, f, indent=2)
        for name in metadata_df.loc[result, "name"]:
            copy_symlink(videos_dir / f"{name}.mp4", dest_videos_dir / f"{name}.mp4")
            copy_symlink(poses_csv_dir / f"{name}.csv", dest_poses_csv_dir / f"{name}.csv", allow_regular=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
