"""Sample strain-balanced stratified training sets across multiple splits.

Reads ``metadata.csv`` from *dataset_dir*, separates records by strain
(DO / B6), and draws *n_samples* records per split while balancing the
two strains equally. Rare strata (fewer than *min_count* members) are
included unconditionally, and the remaining budget is filled via
``StratifiedShuffleSplit``. Each split is exported to a numbered
sub-directory with symlinked video and pose-CSV files and a
``sample_params.json`` manifest.

Usage::

    python sample_combined_kpms_training_set.py \\
        --dataset_dir <path_to_dataset_dir> \\
        --export_name_base <prefix> \\
        --n_samples <int> \\
        --n_splits <int> \\
        [--categorical_vars sex diet] \\
        [--strat_vars_bins age=5] \\
        [--base_seed <int>] \\
        [--min_count <int>]
"""

import argparse
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List, Mapping, Sequence


def _make_strata(
    df: pd.DataFrame,
    strat_vars_bins: Mapping[str, int],
    categorical_vars: Sequence[str],
) -> pd.Series:
    """Build a joint-stratum identifier for each row.

    Bins numeric columns and casts categorical columns to strings, then
    concatenates them with ``"-"`` as separator.

    Args:
        df: Input dataframe.
        strat_vars_bins: Mapping of numeric column names to bin counts.
        categorical_vars: Categorical column names.

    Returns:
        A string Series of joint-stratum identifiers.
    """
    tmp = df.copy()
    for col, n_bins in strat_vars_bins.items():
        tmp[f"__bin_{col}"] = pd.cut(
            tmp[col], bins=n_bins, labels=False, include_lowest=True
        )
    for cat in categorical_vars:
        tmp[f"__bin_{cat}"] = tmp[cat].astype(str)
    cols = [f"__bin_{c}" for c in strat_vars_bins] + [
        f"__bin_{c}" for c in categorical_vars
    ]
    return tmp[cols].astype(str).agg("-".join, axis=1)


def _stratified_sample(
    df: pd.DataFrame,
    k: int,
    seed: int,
    strat_vars_bins: Mapping[str, int],
    categorical_vars: Sequence[str],
    min_count: int = 2,
) -> List[int]:
    """Draw *k* indices from *df* with stratification and rare-stratum handling.

    Strata with fewer than *min_count* members are included first.  The
    remaining budget is filled via ``StratifiedShuffleSplit`` over
    eligible (non-rare) rows.  If stratified sampling fails, a random
    draw is used as a fallback.

    Args:
        df: Subset dataframe to sample from.
        k: Number of indices to draw.
        seed: Random seed for reproducibility.
        strat_vars_bins: Mapping of numeric column names to bin counts.
        categorical_vars: Categorical column names.
        min_count: Strata with fewer members than this are treated as
            rare and included unconditionally.

    Returns:
        List of integer indices into *df*.
    """
    if k <= 0 or df.empty:
        return []

    rng = np.random.RandomState(seed)
    strata = _make_strata(df, strat_vars_bins, categorical_vars)
    counts = strata.value_counts()
    rare_mask = strata.map(lambda x: counts[x] < min_count)
    rare_idx = df[rare_mask].index.tolist()
    take_rare = rare_idx[: min(k, len(rare_idx))]

    remaining_df = df.drop(index=take_rare)
    remaining_k = k - len(take_rare)
    if remaining_k <= 0:
        return take_rare[:k]
    if remaining_df.empty:
        return take_rare

    strata_rem = _make_strata(remaining_df, strat_vars_bins, categorical_vars)
    rem_counts = strata_rem.value_counts()
    ok_mask = strata_rem.map(lambda x: rem_counts[x] >= min_count)
    eligible = remaining_df[ok_mask]

    if len(eligible) >= remaining_k:
        try:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=remaining_k, random_state=seed
            )
            idx, _ = next(
                sss.split(
                    eligible, _make_strata(eligible, strat_vars_bins, categorical_vars)
                )
            )
            return take_rare + eligible.index[idx].tolist()
        except ValueError:
            pass

    pool = remaining_df.index.difference(take_rare)
    if len(pool) <= remaining_k:
        return take_rare + list(pool)
    fill = rng.choice(pool, size=remaining_k, replace=False).tolist()
    return take_rare + fill


def _copy_symlink(src: Path, dst: Path, *, allow_regular: bool = False):
    """Create a symlink at *dst* pointing to the resolved target of *src*.

    Args:
        src: Source path (symlink or regular file).
        dst: Destination path for the new symlink.
        allow_regular: When ``True``, accept regular files as *src*.

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
    export_name_base: str,
    categorical_vars: Sequence[str],
    strat_vars_bins: Mapping[str, int],
    n_samples: int,
    base_seed: int,
    n_splits: int,
    min_count: int,
):
    """Generate *n_splits* strain-balanced stratified training sets.

    For each split, records are divided by strain (DO / B6) and sampled
    in roughly equal proportions. The selected video and pose-CSV files
    are symlinked into numbered export directories under *dataset_dir*.

    Args:
        dataset_dir: Dataset directory containing ``metadata.csv``,
            ``videos/``, and ``poses_csv/``.
        export_name_base: Prefix for export sub-directories (e.g.
            ``"kpms_training_set_150_"`` produces
            ``kpms_training_set_150_1``, ``kpms_training_set_150_2``, ...).
        categorical_vars: Categorical column names used for stratification.
        strat_vars_bins: Mapping of numeric column names to bin counts.
        n_samples: Number of records per split.
        base_seed: Base random seed; each split uses ``base_seed + split_index``.
        n_splits: Number of independent training sets to generate.
        min_count: Rare-stratum threshold passed to ``_stratified_sample``.

    Raises:
        ValueError: If there are not enough rows to fill *n_samples*.
    """
    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    df = metadata_df.copy()

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

        idx_do = _stratified_sample(
            df_do, take_do, seed, strat_vars_bins, categorical_vars, min_count
        )
        idx_b6 = _stratified_sample(
            df_b6, take_b6, seed + 100000, strat_vars_bins, categorical_vars, min_count
        )
        result = idx_do + idx_b6
        if len(result) != n_samples:
            raise ValueError(
                f"Sample size mismatch for split {s}: got {len(result)}, expected {n_samples}."
            )

        export_dir = dataset_dir / f"{export_name_base}{s}"
        dest_videos_dir = export_dir / "videos"
        dest_poses_csv_dir = export_dir / "poses_csv"
        export_dir.mkdir(exist_ok=True)
        dest_videos_dir.mkdir(exist_ok=True)
        dest_poses_csv_dir.mkdir(exist_ok=True)

        with (export_dir / "sample_params.json").open("w") as f:
            json.dump(
                {
                    "dataset_dir": str(dataset_dir),
                    "export_dir": str(export_dir),
                    "categorical_vars": list(categorical_vars),
                    "strat_vars_bins": dict(strat_vars_bins),
                    "n_samples": n_samples,
                    "seed": seed,
                    "split": s,
                    "per_strain_counts": {"DO": len(idx_do), "B6": len(idx_b6)},
                },
                f,
                indent=2,
            )

        print(
            f"split {s}: exporting {len(result)} samples (DO={len(idx_do)}, B6={len(idx_b6)})"
        )
        for name in metadata_df.loc[result, "name"]:
            _copy_symlink(videos_dir / f"{name}.mp4", dest_videos_dir / f"{name}.mp4")
            _copy_symlink(
                poses_csv_dir / f"{name}.csv",
                dest_poses_csv_dir / f"{name}.csv",
                allow_regular=True,
            )

    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate strain-balanced stratified KPMS training sets.",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing metadata.csv, videos/, poses_csv/",
    )
    parser.add_argument(
        "--export_name_base",
        type=str,
        required=True,
        help="Prefix for export sub-directories (e.g. 'kpms_training_set_150_')",
    )
    parser.add_argument(
        "--categorical_vars",
        type=str,
        nargs="+",
        default=["sex", "diet"],
        help="Categorical variables for stratification (default: sex diet)",
    )
    parser.add_argument(
        "--strat_vars_bins",
        type=str,
        nargs="+",
        default=["age=5"],
        help="Numeric stratification variables as name=n_bins (default: age=5)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of records per split",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=623,
        help="Base random seed (default: 623)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        required=True,
        help="Number of independent training sets to generate",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=2,
        help="Rare-stratum threshold for unconditional inclusion (default: 2)",
    )

    args = parser.parse_args()

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
        args.export_name_base,
        args.categorical_vars,
        strat_dict,
        args.n_samples,
        args.base_seed,
        args.n_splits,
        args.min_count,
    )
