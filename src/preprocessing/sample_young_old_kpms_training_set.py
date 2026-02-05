"""Sample strain-balanced, age-filtered training sets (young / old splits).

Extends the strain-balanced sampling approach from
``sample_combined_kpms_training_set.py`` by filtering each strain into
*young* and *old* subsets based on an age quantile threshold before
sampling. For each split, two export directories are created: one for
young animals (bottom *age_frac* quantile) and one for old animals (top
*age_frac* quantile).

Usage::

    python sample_young_old_kpms_training_set.py \\
        --dataset_dir <path_to_dataset_dir> \\
        --export_name_base <prefix> \\
        --n_samples <int> \\
        --n_splits <int> \\
        [--categorical_vars sex diet] \\
        [--strat_vars_bins age=5] \\
        [--base_seed <int>] \\
        [--min_count <int>] \\
        [--age_col <column_name>] \\
        [--age_frac <float>]
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
    age_col: str | None = None,
    age_frac: float | None = None,
    age_group: str = "young",
) -> List[int]:
    """Draw *k* indices with stratification, rare-stratum handling, and optional age filtering.

    When *age_col* and *age_frac* are provided, the dataframe is first
    filtered to the appropriate age tail before sampling.

    Args:
        df: Subset dataframe to sample from.
        k: Number of indices to draw.
        seed: Random seed for reproducibility.
        strat_vars_bins: Mapping of numeric column names to bin counts.
        categorical_vars: Categorical column names.
        min_count: Strata with fewer members than this are treated as
            rare and included unconditionally.
        age_col: Column name containing numeric age values. When
            ``None``, no age filtering is applied.
        age_frac: Fraction of the age distribution to keep. For
            ``"young"`` the bottom *age_frac* quantile is kept; for
            ``"old"`` the top *age_frac* quantile.
        age_group: Either ``"young"`` or ``"old"``.

    Returns:
        List of integer indices into the original *df*.
    """
    if age_col is not None and age_frac is not None and 0 < age_frac < 1:
        s = pd.to_numeric(df[age_col], errors="coerce")
        q = age_frac if age_group == "young" else 1 - age_frac
        t = s.quantile(q)
        df = df[s <= t] if age_group == "young" else df[s >= t]

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
    age_col: str,
    age_frac: float,
):
    """Generate strain-balanced, age-filtered training sets for young and old groups.

    For each split and each age group (``"young"`` / ``"old"``), records are
    divided by strain (DO / B6), filtered to the appropriate age tail, and
    sampled in roughly equal proportions. Results are symlinked into
    ``<dataset_dir>/<export_name_base>_<group>_<split>``.

    Args:
        dataset_dir: Dataset directory containing ``metadata.csv``,
            ``videos/``, and ``poses_csv/``.
        export_name_base: Prefix for export sub-directories.
        categorical_vars: Categorical column names used for stratification.
        strat_vars_bins: Mapping of numeric column names to bin counts.
        n_samples: Number of records per (split, age-group) combination.
        base_seed: Base random seed; each split uses ``base_seed + split_index``.
        n_splits: Number of independent splits to generate.
        min_count: Rare-stratum threshold passed to ``_stratified_sample``.
        age_col: Column name containing numeric age values.
        age_frac: Fraction of the age distribution defining the young/old
            tails (e.g. ``0.25`` keeps the bottom/top 25 %).

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
        for grp in ("young", "old"):
            s_do = pd.to_numeric(df_do[age_col], errors="coerce")
            s_b6 = pd.to_numeric(df_b6[age_col], errors="coerce")
            q = age_frac if grp == "young" else 1 - age_frac
            t_do = s_do.quantile(q)
            t_b6 = s_b6.quantile(q)
            f_do = df_do[s_do <= t_do] if grp == "young" else df_do[s_do >= t_do]
            f_b6 = df_b6[s_b6 <= t_b6] if grp == "young" else df_b6[s_b6 >= t_b6]

            n_do = len(f_do)
            n_b6 = len(f_b6)
            half = n_samples // 2
            rem = n_samples - 2 * half
            take_do = min(half + (1 if rem == 1 and n_do >= n_b6 else 0), n_do)
            take_b6 = min(n_samples - take_do, n_b6)
            if take_do + take_b6 < n_samples:
                raise ValueError(f"Not enough rows for {grp} group in split {s}.")

            idx_do = _stratified_sample(
                df_do,
                take_do,
                seed,
                strat_vars_bins,
                categorical_vars,
                min_count,
                age_col=age_col,
                age_frac=age_frac,
                age_group=grp,
            )
            idx_b6 = _stratified_sample(
                df_b6,
                take_b6,
                seed + 100000,
                strat_vars_bins,
                categorical_vars,
                min_count,
                age_col=age_col,
                age_frac=age_frac,
                age_group=grp,
            )
            result = idx_do + idx_b6
            if len(result) != n_samples:
                raise ValueError(
                    f"Sample size mismatch for {grp} split {s}: "
                    f"got {len(result)}, expected {n_samples}."
                )

            export_dir = dataset_dir / f"{export_name_base}_{grp}_{s}"
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
                        "age_group": grp,
                        "age_frac": age_frac,
                        "age_col": age_col,
                        "per_strain_counts": {"DO": len(idx_do), "B6": len(idx_b6)},
                    },
                    f,
                    indent=2,
                )

            print(
                f"split {s} ({grp}): exporting {len(result)} samples (DO={len(idx_do)}, B6={len(idx_b6)})"
            )
            for name in metadata_df.loc[result, "name"]:
                _copy_symlink(
                    videos_dir / f"{name}.mp4", dest_videos_dir / f"{name}.mp4"
                )
                _copy_symlink(
                    poses_csv_dir / f"{name}.csv",
                    dest_poses_csv_dir / f"{name}.csv",
                    allow_regular=True,
                )

    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate strain-balanced, age-filtered KPMS training sets.",
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
        help="Prefix for export sub-directories",
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
        help="Number of records per (split, age-group) combination",
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
        help="Number of independent splits to generate",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=2,
        help="Rare-stratum threshold for unconditional inclusion (default: 2)",
    )
    parser.add_argument(
        "--age_col",
        type=str,
        default="age",
        help="Column name for age values (default: age)",
    )
    parser.add_argument(
        "--age_frac",
        type=float,
        default=0.25,
        help="Quantile fraction defining young/old tails (default: 0.25)",
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
        args.age_col,
        args.age_frac,
    )
