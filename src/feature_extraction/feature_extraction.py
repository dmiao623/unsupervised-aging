import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell
def _():
    import types, param
    if not hasattr(param, "reactive"):
        param.reactive = types.SimpleNamespace(rx=param.Parameterized)
    return


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import operator
    import os
    import pandas as pd

    from functools import reduce
    from pathlib import Path
    from typing import Any, Dict, Mapping, Optional, Sequence

    import keypoint_moseq as kpms
    return (
        Any,
        Dict,
        Mapping,
        Optional,
        Sequence,
        json,
        kpms,
        mo,
        np,
        operator,
        pd,
        reduce,
    )


@app.cell
def _():
    # unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])

    # project_name  = "2025-07-03_kpms-v2"
    # model_name    = "2025-07-07_model-2"
    # kpms_dir      = unsupervised_aging_dir / "data/kpms_projects"
    # dataset_dir   = unsupervised_aging_dir / "data/datasets/nature-aging_634/"
    # poses_csv_dir = dataset_dir / "poses_csv"

    # supervised_features_path = unsupervised_aging_dir / "data/archive/B6DO_video.csv"
    # adj_metadata_path = unsupervised_aging_dir / "data/adj_metadata_sheets/nature-aging_634_adj_metadata.csv"

    # project_dir = kpms_dir / project_name
    return


@app.cell
def _():
    # unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])

    # project_name  = "2025-07-16_kpms-v3"
    # model_name    = "2025-07-16_model-4"
    # kpms_dir      = unsupervised_aging_dir / "data/kpms_projects"
    # dataset_dir   = unsupervised_aging_dir / "data/datasets/geroscience_492/"
    # poses_csv_dir = dataset_dir / "poses_csv"

    # supervised_features_path = unsupervised_aging_dir / "data/archive/B6DO_video.csv"
    # adj_metadata_path = unsupervised_aging_dir / "data/adj_metadata_sheets/geroscience_492_adj_metadata.csv"

    # project_dir = kpms_dir / project_name
    return


@app.cell
def _():
    # unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])

    # project_name  = "2025-07-20_kpms-v4_150"
    # model_name    = "2025-07-20_model-1"
    # kpms_dir      = unsupervised_aging_dir / "data/kpms_projects"
    # dataset_dir   = unsupervised_aging_dir / "data/datasets/combined_1126/"
    # poses_csv_dir = dataset_dir / "poses_csv"

    # supervised_features_path = unsupervised_aging_dir / "data/archive/B6DO_video.csv"
    # adj_metadata_path = unsupervised_aging_dir / "data/adj_metadata_sheets/combined_1126_adj_metadata.csv"

    # project_dir = kpms_dir / project_name
    return


@app.cell
def _(kpms, model_name, project_dir):
    results = kpms.load_results(project_dir, model_name)
    return (results,)


@app.cell
def _(results):
    sequences = [pose_dict["syllable"] for pose_dict in results.values()]
    unique_syllables = sorted({s for seq in sequences for s in seq})
    print(len(unique_syllables), max(unique_syllables))
    return


@app.cell
def _(Dict, Sequence, mo, np, results):
    def _get_latent_embedding_statistics() -> Dict[str, Sequence]:
        stats = []
        for _, info in mo.status.progress_bar(results.items()):
            latent_embeddings = info["latent_state"]

            means   = latent_embeddings.mean(axis=0)
            medians = np.median(latent_embeddings, axis=0)
            stds    = latent_embeddings.std(axis=0, ddof=0)

            features = np.concatenate((means, medians, stds))
            stats.append(features)

        trans = list(map(list, zip(*stats)))
        feature_len = len(trans)
        assert feature_len % 3 == 0

        ret = {}
        for i in range(feature_len):
            label = ("mean" if i < feature_len // 3 else
                     "median" if i < 2 * feature_len // 3 else "std")
            ret[f"latent_embedding_{label}_{i % (feature_len // 3)}"] = trans[i]
        return ret

    latent_embedding_statistics = _get_latent_embedding_statistics()
    return (latent_embedding_statistics,)


@app.cell
def _(Dict, Sequence, mo, np, results):
    def _get_syllable_frequency_statistics(th: float = 0.0) -> Dict[str, Sequence[int]]:
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
        for _, info in mo.status.progress_bar(results.items()):
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

    syllable_frequency_statistics = _get_syllable_frequency_statistics()
    return (syllable_frequency_statistics,)


@app.cell
def _(Dict, Mapping, Optional, Sequence, mo, np, results):
    def _get_metasyllable_transition_matrix(
        grouped_syllables: Optional[Mapping[str, Sequence[int]]] = None,
        *,
        ignore_unknown: bool = False,
        include_frequencies: bool = True,
    ) -> Dict[str, Sequence[float]]:
        if grouped_syllables is None:
            grouped_syllables = {}

        sequences = [info["syllable"] for info in results.values()]
        vocab_size = max(s for seq in sequences for s in seq) + 1
        all_indices = set(range(vocab_size))

        seen = set()
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

        names      = list(grouped_syllables.keys())
        idx_sets   = [set(grouped_syllables[n]) for n in names]
        g          = len(names)
        feats      = {f"transition_matrix_{a}_{b}": [] for a in names for b in names if a != b}
        if include_frequencies:
            feats.update({f"metasyllable_frequency_{n}": [] for n in names})

        idx_to_group = {}
        for gi, s in enumerate(idx_sets):
            for idx in s:
                idx_to_group[idx] = gi

        for _, info in mo.status.progress_bar(results.items()):
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


    # nature-aging_634
    # _metasyllable_groupings = {
    #     "kpms_dendrogram_0": [0, 2, 10, 54, 35, 9, 30, 16, 26, 20, 6, 15],
    #     "kpms_dendrogram_1": [24, 42, 52, 50, 48, 57, 33, 38, 60, 12, 58, 22, 43],
    #     "kpms_dendrogram_2": [19, 59, 1, 3, 14, 18, 34, 5, 7, 46, 40, 4, 11, 45],
    #     "kpms_dendrogram_3": [13, 8, 17, 39, 51, 21, 36, 61, 31, 49, 28, 44, 55, 37, 25, 32, 27, 56],
    #     "kpms_dendrogram_4": [53, 62, 29, 41, 23, 47]
    # }

    # geroscience_492
    # _metasyllable_groupings = {
    #     "kpms_dendogram_0": [12, 20, 28, 14, 26],
    #     "kpms_dendogram_1": [33, 23, 39, 13, 3, 11, 18],
    #     "kpms_dendogram_2": [24, 9, 6, 25, 15, 21, 16, 35, 2, 10, 17],
    #     "kpms_dendogram_3": [4, 34, 22, 30, 27, 29, 32, 19],
    #     "kpms_dendogram_4": [5, 7, 8, 43, 55, 0, 1, 31],
    # }

    # combined_1126
    # _metasyllable_groupings = {
    #     "kpms_dendrogram_0": [41, 11, 23, 39, 22, 37, 28, 32, 5, 34, 1, 31, 20, 13, 25],
    #     "kpms_dendrogram_1": [45, 46],
    #     "kpms_dendrogram_2": [44, 50, 4, 2, 18, 53, 24, 8, 35, 14, 15, 10, 17, 26, 30, 7, 43, 9, 42, 48, 6, 29],
    #     "kpms_dendrogram_3": [47, 27, 36],
    #     "kpms_dendrogram_4": [40, 21, 12, 33, 16, 0, 3, 19, 38],
    # }

    kpms_dendrogram_metasyllable_transition_matrix = _get_metasyllable_transition_matrix(_metasyllable_groupings, ignore_unknown=True, include_frequencies=False)
    return (kpms_dendrogram_metasyllable_transition_matrix,)


@app.cell
def _(
    Any,
    Dict,
    Sequence,
    kpms_dendrogram_metasyllable_transition_matrix,
    latent_embedding_statistics,
    operator,
    pd,
    reduce,
    results,
    syllable_frequency_statistics,
):
    def _merge_features(
        features: Sequence[Dict[str, Sequence[Any]]]
    ) -> pd.DataFrame:
        names = list(map(lambda path: path.removesuffix(".csv"), results.keys()))
        merged_features = reduce(operator.or_, [{"name": names}] + features, {})
        return pd.DataFrame(merged_features)

    unsupervised_features_df = _merge_features([
        latent_embedding_statistics, 
        syllable_frequency_statistics,
        kpms_dendrogram_metasyllable_transition_matrix
    ])
    unsupervised_features_df
    return (unsupervised_features_df,)


@app.cell
def _(adj_metadata_path, pd, unsupervised_features_df):
    ### merge with metadata matrix (should have same number of rows)

    metadata_df = pd.read_csv(adj_metadata_path)
    metadata_unsupervised_features_df = metadata_df.merge(unsupervised_features_df, on="name", how="inner")
    metadata_unsupervised_features_df
    return (metadata_unsupervised_features_df,)


@app.cell
def _(metadata_unsupervised_features_df, pd, supervised_features_path):
    supervised_features_df = pd.read_csv(supervised_features_path)
    supervised_columns = [
        col for col in supervised_features_df.columns if col not in ["NetworkFilename", "PoseFilename", "Batch", "Tester", "AgeGroup", "MouseID", "Strain", "Diet", "Weight", "Sex", "AgeW", "AgeAtVid", "CFI_norm", "FLL", "score"]
    ]

    _filename_to_name = {
        row["name"].split("__")[-1]: row["name"] for _, row in metadata_unsupervised_features_df.iterrows()
    }
    _names = []
    for _, row in supervised_features_df.iterrows():
        if row["PoseFilename"].startswith("/"):
            _names.append(_filename_to_name.get(row["PoseFilename"][1:], pd.NA))
        else:
            _names.append(row["NetworkFilename"].removesuffix(".avi").replace("/", "__"))
    supervised_features_df["name"] = _names
    supervised_features_df = supervised_features_df.drop_duplicates()

    ### drop only duplicated row (unused)
    _mask = (
        (supervised_features_df["name"] == "LL3-B2B__2020-01-02_SPD__LL3-4_AgedB6-0842")
        & (supervised_features_df["AgeW"] == 20)
    )
    supervised_features_df.drop(supervised_features_df[_mask].index, inplace=True)

    supervised_features_df = supervised_features_df[["name"] + supervised_columns].copy()

    supervised_features_df
    return supervised_columns, supervised_features_df


@app.cell
def _(metadata_unsupervised_features_df, supervised_features_df):
    features_df = metadata_unsupervised_features_df.merge(
        supervised_features_df,
        on="name",
        how="inner"
    )
    features_df
    return (features_df,)


@app.cell
def _(
    kpms_dendrogram_metasyllable_transition_matrix,
    latent_embedding_statistics,
    supervised_columns,
    syllable_frequency_statistics,
):
    _all_unsupervised_columns = (
        list(latent_embedding_statistics.keys()) + 
        list(syllable_frequency_statistics.keys()) +
        list(kpms_dendrogram_metasyllable_transition_matrix.keys())
    )

    Xcats = {
        "unsupervised": _all_unsupervised_columns,
        "supervised": supervised_columns,
        "all": _all_unsupervised_columns + supervised_columns
    }
    return (Xcats,)


@app.cell
def _(features_df):
    # features_df.drop("fll", axis=1, inplace=True)
    rows_with_na = features_df[features_df.isna().any(axis=1)].copy()

    rows_with_na["na_cols"] = (
        rows_with_na
        .isna()
        .apply(lambda r: [c for c, is_na in r.items() if is_na], axis=1)
    )

    rows_with_na["na_cols"]
    return


@app.cell
def _(
    Xcats,
    dataset_dir,
    features_df,
    json,
    model_name,
    project_name,
    unsupervised_aging_dir,
):
    _uid = f"{dataset_dir.name}__{project_name}__{model_name}"

    from datetime import datetime
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%Y-%m-%d")

    feature_matrix_output_dir  = unsupervised_aging_dir / "data/feature_matrices"
    feature_matrix_output_path = feature_matrix_output_dir / f"{formatted_date}_feature-matrix__{_uid}.csv"
    xcats_output_path          = feature_matrix_output_dir / f"{formatted_date}_xcats__{_uid}.json"

    features_df.to_csv(feature_matrix_output_path)
    with xcats_output_path.open("w") as f:
        json.dump(Xcats, f, indent=2)

    print(f"wrote feature matrix to  `{feature_matrix_output_path}`")
    print(f"wrote X category JSON to `{xcats_output_path}`")
    print(f"X category keys: {Xcats.keys()}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
