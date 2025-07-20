import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


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
        Path,
        Sequence,
        json,
        kpms,
        mo,
        np,
        operator,
        os,
        pd,
        reduce,
    )


@app.cell
def _(Path, os):
    project_name  = "2025-07-16_kpms-v3"
    model_name    = "2025-07-16_model-4"
    kpms_dir      = Path(os.environ["UNSUPERVISED_AGING"] + "/data/kpms_projects")
    dataset_dir   = Path(os.environ["UNSUPERVISED_AGING"] + "/data/datasets/geroscience_492/")
    poses_csv_dir = dataset_dir / "poses_csv"

    supervised_features_path = Path(os.environ["UNSUPERVISED_AGING"]) / "data/archive/B6DO_video.csv"

    project_dir = kpms_dir / project_name
    return dataset_dir, model_name, project_dir, supervised_features_path


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

    grouped_syllable_transition_matrix = _get_metasyllable_transition_matrix({
        "walking":    [5, 13],
        "turning":    [4, 9, 17, 19],
        "rearing":    [0, 2, 7, 8, 10, 11, 16, 21],
        "stationary": [1, 3, 6, 12, 15, 18, 22], 
    })
    return


@app.cell
def _(
    Any,
    Dict,
    Sequence,
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
        # grouped_syllable_transition_matrix,
    ])
    unsupervised_features_df
    return (unsupervised_features_df,)


@app.cell
def _(dataset_dir, pd, unsupervised_features_df):
    ### merge with metadata matrix (should have same number of rows)

    metadata_path = dataset_dir / "metadata.csv"
    metadata_df = pd.read_csv(metadata_path)
    metadata_unsupervised_features_df = metadata_df.merge(unsupervised_features_df, on="name", how="inner")
    metadata_unsupervised_features_df
    return (metadata_unsupervised_features_df,)


@app.cell
def _(metadata_unsupervised_features_df):
    left_key = ["mouse_id", "sex", "fi"]

    dups_left = (
        metadata_unsupervised_features_df[
            metadata_unsupervised_features_df.duplicated(left_key, keep=False)
        ]
        .sort_values(left_key)            # nice, readable order
    )

    dups_left
    return


@app.cell
def _(metadata_unsupervised_features_df, pd, supervised_features_path):
    ### merge with supervised feature matrix

    supervised_features_df = pd.read_csv(supervised_features_path)
    supervised_columns = [
        col for col in supervised_features_df.columns if col not in ["NetworkFilename", "PoseFilename", "Batch", "Tester", "AgeGroup", "MouseID", "Strain", "Diet", "Weight", "Sex", "AgeW", "AgeAtVid", "CFI_norm", "FLL", "score"]
    ]
    supervised_features_df["name"] = (
        supervised_features_df["NetworkFilename"]
          .str.replace("/", "__")
          .str.replace(r"\.avi$", "", regex=True)
    )

    matched_features_df = metadata_unsupervised_features_df.merge(
        supervised_features_df, 
        on="name",
        how="inner"
    )
    matched_features_df = (matched_features_df.loc[matched_features_df["age"].sub(matched_features_df["AgeAtVid"]).abs().le(1)])
    matched_features_df = matched_features_df[list(metadata_unsupervised_features_df.columns) + supervised_columns].copy()
    matched_features_df
    return matched_features_df, supervised_columns, supervised_features_df


@app.cell
def _(metadata_unsupervised_features_df, supervised_features_df):
    names_unsup = set(metadata_unsupervised_features_df['name'])
    names_sup   = set(supervised_features_df['name'])

    missing_from_sup = names_unsup - names_sup

    unmatched_metadata_unsupervised_features_df = metadata_unsupervised_features_df[
        metadata_unsupervised_features_df["name"].isin(missing_from_sup)
    ].copy()
    unmatched_metadata_unsupervised_features_df["NetworkFilename"] = (
        "/" + unmatched_metadata_unsupervised_features_df["name"]
            .str.split("__")
            .str[-1]
            .astype(str) + ".avi"
    )
    print(unmatched_metadata_unsupervised_features_df["NetworkFilename"])

    unmatched_features_df = unmatched_metadata_unsupervised_features_df.merge(
        supervised_features_df, 
        on="NetworkFilename",
        how="inner"
    )
    unmatched_features_df
    return (unmatched_features_df,)


@app.cell
def _(matched_features_df, pd, unmatched_features_df):
    features_df = pd.concat((matched_features_df, unmatched_features_df), axis=0)
    features_df
    return (features_df,)


@app.cell
def _(
    latent_embedding_statistics,
    supervised_columns,
    syllable_frequency_statistics,
):
    _all_unsupervised_columns = (
        list(latent_embedding_statistics.keys()) + 
        list(syllable_frequency_statistics.keys())
        # list(grouped_syllable_transition_matrix.keys())
    )

    Xcats = {
        "kpms-v2_all": _all_unsupervised_columns,
        # "kpms-v2_nonmeta": (
        #    list(latent_embedding_statistics.keys()) + 
        #    list(syllable_frequency_statistics.keys())
        #),
        "supervised": supervised_columns,
        "all": _all_unsupervised_columns + supervised_columns
    }
    return (Xcats,)


@app.cell
def _(Path, Xcats, features_df, json, os):
    feature_matrix_output_dir = Path(os.environ["UNSUPERVISED_AGING"] + "/data/feature_matrices")
    feature_matrix_output_path = feature_matrix_output_dir / "2025-07-16_kpms-v3-supervised_feature-matrix.csv"
    xcats_output_path = feature_matrix_output_dir / "2025-07-16_kpms-v3-supervised_xcats.json"

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
