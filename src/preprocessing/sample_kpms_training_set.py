"""Sample a stratified subset and transfer files to an export directory.

Reads ``metadata.csv`` from *dataset_dir*, bins numeric stratification
variables, and uses ``StratifiedShuffleSplit`` to draw *n_samples* records
while approximately preserving the joint distribution of the provided
categorical and binned-numeric variables. Selected records' video (``.mp4``)
and pose-estimation CSV (``.csv``) files are symlinked into
``<export_dir>/videos`` and ``<export_dir>/poses_csv``. Sampling parameters
are written to ``sample_params.json`` in *export_dir*. Optionally, a
comparison figure of population vs. sampled counts is saved.

SLURM Template:
    scripts/templates/sample_kpms_training_set.sh

Usage::

    python sample_kpms_training_set.py \\
        --dataset_dir <path_to_dataset_dir> \\
        --export_dir <path_to_export_dir> \\
        --categorical_vars sex diet \\
        --strat_vars_bins age=5 \\
        --n_samples 200 \\
        --seed 623 \\
        [--figure_path <path_to_output_figure>]
"""

import argparse
import json
import os
import pandas as pd

from pathlib import Path
from plotnine import (
    ggplot,
    aes,
    element_text,
    geom_col,
    facet_wrap,
    labs,
    theme,
    theme_bw,
    position_dodge,
)
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Mapping, Optional, Sequence


def _copy_symlink(src: Path, dst: Path, *, allow_regular: bool = False):
    """Create a symlink at *dst* pointing to the target of *src*.

    If *src* is itself a symlink, the new link points to the resolved
    target. If *allow_regular* is ``True`` and *src* is a regular file,
    the link points directly to *src*.

    Args:
        src: Source path (symlink or regular file).
        dst: Destination path for the new symlink.
        allow_regular: When ``True``, accept regular files as *src* in
            addition to symlinks.

    Raises:
        ValueError: If *src* is neither a symlink nor (when permitted) a
            regular file.
        FileNotFoundError: If the resolved target does not exist.
    """
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


def main(
    dataset_dir: Path,
    export_dir: Path,
    categorical_vars: Sequence[str],
    strat_vars_bins: Mapping[str, int],
    n_samples: int,
    seed: int,
    figure_path: Optional[Path],
):
    """Draw a stratified sample and symlink associated files to *export_dir*.

    Bins numeric columns according to *strat_vars_bins*, combines them with
    *categorical_vars* into joint strata, performs a
    ``StratifiedShuffleSplit``, and symlinks the selected video and pose CSV
    files. A ``sample_params.json`` file is written to *export_dir*.

    Args:
        dataset_dir: Dataset directory containing ``metadata.csv``,
            ``videos/``, and ``poses_csv/``.
        export_dir: Directory where the sampled subset will be exported.
        categorical_vars: Categorical column names used for stratification.
        strat_vars_bins: Mapping of numeric column names to the number of
            bins to use for stratification.
        n_samples: Number of records to sample.
        seed: Random seed for reproducibility.
        figure_path: If provided, a population-vs-sample comparison figure
            is saved to this path.

    Raises:
        ValueError: If the resulting sample size does not match *n_samples*.
    """
    dataset_dir = Path(dataset_dir)
    export_dir = Path(export_dir)
    if figure_path is not None:
        figure_path = Path(figure_path)

    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    df = metadata_df.copy()

    for col, n_bins in strat_vars_bins.items():
        df[f"__bin_{col}"] = pd.cut(
            df[col], bins=n_bins, labels=False, include_lowest=True
        )
    for cat in categorical_vars:
        df[f"__bin_{cat}"] = df[cat].astype(str)

    all_bins = [f"__bin_{c}" for c in list(strat_vars_bins) + list(categorical_vars)]
    strata = df[all_bins].astype(str).agg("-".join, axis=1)

    pop_counts = strata.value_counts(sort=False).sort_index()
    pop_dist_df = pop_counts.reset_index()
    pop_dist_df.columns = ["stratum", "population_count"]

    print("\n--- STRATA POPULATION DISTRIBUTION ---")
    print(pop_dist_df.to_string(index=False))
    print("--------------------------------------\n")

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(sss.split(df, strata))
    result = df.index[train_idx].tolist()
    if len(result) != n_samples:
        raise ValueError("Sample size mismatch.")

    export_dir.mkdir(exist_ok=True)
    (export_dir / "videos").mkdir(exist_ok=True)
    (export_dir / "poses_csv").mkdir(exist_ok=True)
    with (export_dir / "sample_params.json").open("w") as f:
        json.dump(
            {
                "dataset_dir": str(dataset_dir),
                "export_dir": str(export_dir),
                "categorical_vars": categorical_vars,
                "strat_vars_bins": strat_vars_bins,
                "n_samples": n_samples,
                "seed": seed,
            },
            f,
            indent=2,
        )

    videos_dir = dataset_dir / "videos"
    poses_csv_dir = dataset_dir / "poses_csv"
    selected_names = df.loc[result, "name"].tolist()

    print(f"Beginning export of {len(selected_names)} samples.")
    for name in selected_names:
        _copy_symlink(videos_dir / f"{name}.mp4", export_dir / "videos" / f"{name}.mp4")
        _copy_symlink(
            poses_csv_dir / f"{name}.csv",
            export_dir / "poses_csv" / f"{name}.csv",
            allow_regular=True,
        )

    if figure_path is not None:
        plot_frames = []
        all_vars = list(strat_vars_bins.keys()) + list(categorical_vars)
        for col in all_vars:
            pop = (
                df[f"__bin_{col}"]
                .value_counts(sort=False)
                .rename_axis("bin")
                .reset_index(name="count")
            )
            pop["variable"] = col
            pop["set"] = "population"
            samp = (
                df.loc[result, f"__bin_{col}"]
                .value_counts(sort=False)
                .rename_axis("bin")
                .reset_index(name="count")
            )
            samp["variable"] = col
            samp["set"] = "sample"
            plot_frames.extend([pop, samp])

        joint_cols = [f"__bin_{v}" for v in all_vars]
        df["__joint"] = df[joint_cols].astype(str).agg("-".join, axis=1)
        pop_joint = (
            df["__joint"]
            .value_counts(sort=False)
            .rename_axis("bin")
            .reset_index(name="count")
        )
        pop_joint["variable"] = "joint"
        pop_joint["set"] = "population"
        samp_joint = (
            df.loc[result, "__joint"]
            .value_counts(sort=False)
            .rename_axis("bin")
            .reset_index(name="count")
        )
        samp_joint["variable"] = "joint"
        samp_joint["set"] = "sample"
        plot_frames.extend([pop_joint, samp_joint])

        plot_df = pd.concat(plot_frames, ignore_index=True)
        plot_df["bin"] = plot_df["bin"].astype(str)
        p = (
            ggplot(plot_df, aes("bin", "count", fill="set"))
            + geom_col(position=position_dodge(width=0.8))
            + facet_wrap("~variable", scales="free_x")
            + labs(
                x="Bin / Joint-stratum ID",
                y="Count",
                fill="",
                title="Population vs. Sample counts",
            )
            + theme_bw()
            + theme(axis_text_x=element_text(angle=90, hjust=1))
        )
        p.save(filename=str(figure_path), verbose=False)
        print(f"Figure saved to {figure_path}.")

    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample a stratified subset and transfer video/pose CSV files.",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing metadata.csv, videos/, poses_csv/",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Path to export directory for the sampled subset",
    )
    parser.add_argument(
        "--categorical_vars",
        type=str,
        nargs="+",
        required=True,
        help="Categorical variables for stratification",
    )
    parser.add_argument(
        "--strat_vars_bins",
        type=str,
        nargs="+",
        required=True,
        help="Numeric stratification variables specified as name=n_bins",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of records to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=623,
        help="Random seed (default: 623)",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default=None,
        help="Optional path to save population-vs-sample comparison figure",
    )

    args = parser.parse_args()

    if args.export_dir is None or args.export_dir == "":
        args.export_dir = str(
            Path(args.dataset_dir) / f"kpms_training_set_{args.n_samples}"
        )

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    strat_dict = {}
    for item in args.strat_vars_bins:
        if "=" not in item:
            raise ValueError("Invalid --strat_vars_bins entry. Use name=n_bins.")
        k, v = item.split("=", 1)
        strat_dict[k] = int(v)

    main(
        Path(args.dataset_dir),
        Path(args.export_dir),
        args.categorical_vars,
        strat_dict,
        args.n_samples,
        args.seed,
        Path(args.figure_path) if args.figure_path else None,
    )
