"""Extract unsupervised and supervised feature matrices from KPMS results.

Computes latent-embedding statistics, syllable frequency distributions,
and (optionally) meta-syllable transition matrices from trained KPMS
model results. Features are merged with metadata and optionally with
supervised features, then exported as a CSV feature matrix and a JSON
file describing the feature categories.

Usage::

    python feature_extraction.py \\
        --project_name <project_name> \\
        --model_name <model_name> \\
        --kpms_dir <path_to_kpms_projects> \\
        --dataset_dir <path_to_dataset> \\
        --adj_metadata_path <path_to_metadata_csv> \\
        --output_dir <path_to_output> \\
        [--supervised_features_path <path_to_supervised_csv>] \\
        [--metasyllable_groupings_path <path_to_groupings_json>]
"""

import argparse
import json
import operator
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import keypoint_moseq as kpms
from tqdm import tqdm

# Columns in the supervised features CSV that are metadata (not features).
SUPERVISED_METADATA_COLUMNS = {
    "NetworkFilename",
    "PoseFilename",
    "Batch",
    "Tester",
    "AgeGroup",
    "MouseID",
    "Strain",
    "Diet",
    "Weight",
    "Sex",
    "AgeW",
    "AgeAtVid",
    "CFI_norm",
    "FLL",
    "score",
}


def get_latent_embedding_statistics(
    results: Dict[str, Any],
) -> Dict[str, list]:
    """Compute mean, median, and std of latent embeddings per video.

    For each video the latent-state matrix is summarized along the time
    axis, producing one scalar per latent dimension for each of the three
    statistics.

    Args:
        results: KPMS results dict keyed by video name.

    Returns:
        Dict mapping feature names (e.g. ``latent_embedding_mean_0``) to
        lists of per-video values.
    """
    stats = []
    for _, info in tqdm(results.items(), desc="latent embeddings"):
        latent_embeddings = info["latent_state"]
        means = latent_embeddings.mean(axis=0)
        medians = np.median(latent_embeddings, axis=0)
        stds = latent_embeddings.std(axis=0, ddof=0)
        stats.append(np.concatenate((means, medians, stds)))

    trans = list(map(list, zip(*stats)))
    feature_len = len(trans)
    assert feature_len % 3 == 0

    ret = {}
    for i in range(feature_len):
        label = (
            "mean"
            if i < feature_len // 3
            else "median" if i < 2 * feature_len // 3 else "std"
        )
        ret[f"latent_embedding_{label}_{i % (feature_len // 3)}"] = trans[i]
    return ret


def get_syllable_frequency_statistics(
    results: Dict[str, Any],
    th: float = 0.0,
) -> Dict[str, list]:
    """Compute per-video syllable frequency distributions.

    Args:
        results: KPMS results dict keyed by video name.
        th: Minimum global frequency threshold. Syllables whose overall
            fraction falls below this value are dropped.

    Returns:
        Dict mapping feature names (e.g. ``syllable_frequency_3``) to
        lists of per-video frequency values.
    """
    sequences = [info["syllable"] for info in results.values()]
    uniq = sorted({s for seq in sequences for s in seq})
    if th > 0.0:
        global_counts = {s: 0 for s in uniq}
        for seq in sequences:
            for s in seq:
                global_counts[s] += 1
        total = sum(global_counts.values())
        uniq = [s for s in uniq if global_counts[s] / total >= th]

    idx = {s: i for i, s in enumerate(uniq)}
    n = len(uniq)

    freqs_per_video = []
    for _, info in tqdm(results.items(), desc="syllable frequencies"):
        seq = info["syllable"]
        cnt = np.zeros(n, dtype=int)
        for s in seq:
            if s in idx:
                cnt[idx[s]] += 1
        total_tokens = len(seq)
        freqs = cnt / total_tokens if total_tokens else cnt
        freqs_per_video.append(freqs)

    transposed = list(map(list, zip(*freqs_per_video)))
    return {f"syllable_frequency_{s}": transposed[i] for i, s in enumerate(uniq)}


def get_metasyllable_transition_matrix(
    results: Dict[str, Any],
    grouped_syllables: Optional[Mapping[str, Sequence[int]]] = None,
    *,
    ignore_unknown: bool = False,
    include_frequencies: bool = True,
) -> Dict[str, list]:
    """Compute per-video transition matrices between meta-syllable groups.

    Syllable indices are mapped to named groups via *grouped_syllables*.
    For each video, a row-normalized transition matrix (with the diagonal
    zeroed) is computed over the group labels.

    Args:
        results: KPMS results dict keyed by video name.
        grouped_syllables: Mapping from group name to list of syllable
            indices. If ``None``, an empty mapping is used.
        ignore_unknown: If ``False``, syllables not covered by any group
            are collected into an ``"unknown"`` group.
        include_frequencies: If ``True``, per-group frequency features
            are also included.

    Returns:
        Dict mapping feature names to lists of per-video values.

    Raises:
        ValueError: If *grouped_syllables* contains invalid or duplicate
            indices.
    """
    if grouped_syllables is None:
        grouped_syllables = {}

    sequences = [info["syllable"] for info in results.values()]
    vocab_size = max(s for seq in sequences for s in seq) + 1
    all_indices = set(range(vocab_size))

    seen: set[int] = set()
    for name, idxs in grouped_syllables.items():
        bad = set(idxs) - all_indices
        if bad:
            raise ValueError(f"Group '{name}' contains invalid indices {sorted(bad)}.")
        if seen.intersection(idxs):
            raise ValueError("Duplicate indices detected across groups.")
        seen.update(idxs)

    if not ignore_unknown:
        unknown = sorted(all_indices - seen)
        if unknown:
            grouped_syllables = dict(grouped_syllables)
            grouped_syllables["unknown"] = unknown

    names = list(grouped_syllables.keys())
    idx_sets = [set(grouped_syllables[n]) for n in names]
    g = len(names)
    feats: Dict[str, list] = {
        f"transition_matrix_{a}_{b}": [] for a in names for b in names if a != b
    }
    if include_frequencies:
        feats.update({f"metasyllable_frequency_{n}": [] for n in names})

    idx_to_group: Dict[int, int] = {}
    for gi, s in enumerate(idx_sets):
        for idx in s:
            idx_to_group[idx] = gi

    for _, info in tqdm(results.items(), desc="transition matrices"):
        seq = info["syllable"]
        G = np.zeros((g, g), dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            if a in idx_to_group and b in idx_to_group:
                G[idx_to_group[a], idx_to_group[b]] += 1

        np.fill_diagonal(G, 0)
        row_sums = G.sum(axis=1, keepdims=True)
        np.divide(G, row_sums, out=G, where=row_sums != 0)

        for i, ai in enumerate(names):
            for j, bj in enumerate(names):
                if ai != bj:
                    feats[f"transition_matrix_{ai}_{bj}"].append(G[i, j])

        if include_frequencies:
            counts = np.zeros(g, dtype=int)
            for s in seq:
                if s in idx_to_group:
                    counts[idx_to_group[s]] += 1
            total_tokens = len(seq)
            freqs = counts / total_tokens if total_tokens else counts
            for i, name in enumerate(names):
                feats[f"metasyllable_frequency_{name}"].append(freqs[i])

    return feats


def merge_supervised_features(
    metadata_unsupervised_df: pd.DataFrame,
    supervised_features_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge supervised features with the unsupervised feature matrix.

    Reads the supervised features CSV, resolves filename-to-name
    mappings, and performs an inner join on the ``name`` column.

    Args:
        metadata_unsupervised_df: DataFrame with unsupervised features
            and metadata, containing a ``name`` column.
        supervised_features_path: Path to the supervised features CSV.

    Returns:
        A tuple of ``(merged_df, supervised_columns)`` where
        *supervised_columns* lists the feature column names from the
        supervised CSV.
    """
    supervised_features_df = pd.read_csv(supervised_features_path)
    supervised_columns = [
        col
        for col in supervised_features_df.columns
        if col not in SUPERVISED_METADATA_COLUMNS
    ]

    filename_to_name = {
        row["name"].split("__")[-1]: row["name"]
        for _, row in metadata_unsupervised_df.iterrows()
    }

    names = []
    for _, row in supervised_features_df.iterrows():
        if row["PoseFilename"].startswith("/"):
            names.append(filename_to_name.get(row["PoseFilename"][1:], pd.NA))
        else:
            names.append(row["NetworkFilename"].removesuffix(".avi").replace("/", "__"))

    supervised_features_df["name"] = names
    supervised_features_df = supervised_features_df.drop_duplicates()

    # Drop known duplicate row (dataset-specific).
    mask = (
        supervised_features_df["name"] == "LL3-B2B__2020-01-02_SPD__LL3-4_AgedB6-0842"
    ) & (supervised_features_df["AgeW"] == 20)
    supervised_features_df = supervised_features_df.drop(
        supervised_features_df[mask].index,
    )

    supervised_features_df = supervised_features_df[
        ["name"] + supervised_columns
    ].copy()

    merged_df = metadata_unsupervised_df.merge(
        supervised_features_df,
        on="name",
        how="inner",
    )
    return merged_df, supervised_columns


def main(
    project_name: str,
    model_name: str,
    kpms_dir: str,
    dataset_dir: str,
    adj_metadata_path: str,
    output_dir: str,
    supervised_features_path: str | None = None,
    metasyllable_groupings_path: str | None = None,
):
    """Extract features from KPMS results and export as CSV.

    Computes unsupervised features (latent embeddings, syllable
    frequencies, and optionally meta-syllable transitions), merges with
    metadata and optionally supervised features, and writes the feature
    matrix CSV and feature-category JSON.

    Args:
        project_name: Name of the KPMS project (subdirectory of *kpms_dir*).
        model_name: Name of the trained model.
        kpms_dir: Parent directory containing KPMS project directories.
        dataset_dir: Path to the dataset directory (used for output naming).
        adj_metadata_path: Path to the adjusted metadata CSV.
        output_dir: Directory where output files are written.
        supervised_features_path: Optional path to supervised features CSV.
        metasyllable_groupings_path: Optional path to a JSON file mapping
            group names to lists of syllable indices.
    """
    kpms_dir = Path(kpms_dir)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    project_dir = kpms_dir / project_name

    print("--- LOADING RESULTS ---")
    results = kpms.load_results(project_dir, model_name)

    sequences = [info["syllable"] for info in results.values()]
    unique_syllables = sorted({s for seq in sequences for s in seq})
    print(
        f"unique syllables: {len(unique_syllables)}, max index: {max(unique_syllables)}"
    )

    # -- Unsupervised features -------------------------------------------

    print("\n--- COMPUTING LATENT EMBEDDING STATISTICS ---")
    latent_embedding_statistics = get_latent_embedding_statistics(results)

    print("\n--- COMPUTING SYLLABLE FREQUENCIES ---")
    syllable_frequency_statistics = get_syllable_frequency_statistics(results)

    unsupervised_feature_dicts: list[Dict[str, list]] = [
        latent_embedding_statistics,
        syllable_frequency_statistics,
    ]

    metasyllable_transition_features: Dict[str, list] = {}
    if metasyllable_groupings_path is not None:
        print("\n--- COMPUTING METASYLLABLE TRANSITION MATRIX ---")
        with open(metasyllable_groupings_path) as f:
            groupings = json.load(f)
        groupings = {k: [int(i) for i in v] for k, v in groupings.items()}
        metasyllable_transition_features = get_metasyllable_transition_matrix(
            results,
            groupings,
            ignore_unknown=True,
            include_frequencies=False,
        )
        unsupervised_feature_dicts.append(metasyllable_transition_features)

    # -- Merge into DataFrame --------------------------------------------

    print("\n--- MERGING FEATURES ---")
    names = list(map(lambda p: p.removesuffix(".csv"), results.keys()))
    merged = reduce(operator.or_, [{"name": names}] + unsupervised_feature_dicts, {})
    unsupervised_df = pd.DataFrame(merged)

    metadata_df = pd.read_csv(adj_metadata_path)
    features_df = metadata_df.merge(unsupervised_df, on="name", how="inner")

    all_unsupervised_columns = (
        list(latent_embedding_statistics.keys())
        + list(syllable_frequency_statistics.keys())
        + list(metasyllable_transition_features.keys())
    )
    xcats: Dict[str, list[str]] = {"unsupervised": all_unsupervised_columns}

    if supervised_features_path is not None:
        print("\n--- MERGING SUPERVISED FEATURES ---")
        features_df, supervised_columns = merge_supervised_features(
            features_df,
            Path(supervised_features_path),
        )
        xcats["supervised"] = supervised_columns
        xcats["all"] = all_unsupervised_columns + supervised_columns

    # -- Export ----------------------------------------------------------

    uid = f"{dataset_dir.name}__{project_name}__{model_name}"
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix_path = output_dir / f"{today}_feature-matrix__{uid}.csv"
    xcats_path = output_dir / f"{today}_xcats__{uid}.json"

    features_df.to_csv(feature_matrix_path)
    with xcats_path.open("w") as f:
        json.dump(xcats, f, indent=2)

    print(f"\nwrote feature matrix to  `{feature_matrix_path}`")
    print(f"wrote X category JSON to `{xcats_path}`")
    print(f"X category keys: {list(xcats.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract feature matrices from KPMS results.",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the keypoint-MoSeq project",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of keypoint-MoSeq model"
    )
    parser.add_argument(
        "--kpms_dir",
        type=str,
        required=True,
        help="Path of the keypoint-MoSeq project directory",
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--adj_metadata_path",
        type=str,
        required=True,
        help="Path to the adjusted metadata CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write output files to",
    )
    parser.add_argument(
        "--supervised_features_path",
        type=str,
        default=None,
        help="Optional path to supervised features CSV",
    )
    parser.add_argument(
        "--metasyllable_groupings_path",
        type=str,
        default=None,
        help="Optional path to JSON file with meta-syllable groupings",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(
        args.project_name,
        args.model_name,
        args.kpms_dir,
        args.dataset_dir,
        args.adj_metadata_path,
        args.output_dir,
        supervised_features_path=args.supervised_features_path,
        metasyllable_groupings_path=args.metasyllable_groupings_path,
    )
