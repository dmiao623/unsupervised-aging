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
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram

    from pathlib import Path
    from typing import Callable, Dict, Set, Tuple, List, Union

    import keypoint_moseq as kpms
    return (
        Callable,
        Dict,
        List,
        Path,
        Set,
        Tuple,
        Union,
        dendrogram,
        kpms,
        mo,
        np,
        plt,
    )


@app.cell
def _():
    import os
    import sys

    sys.path.append(os.environ["UNSUPERVISED_AGING"] + "/src/kpms_utils")

    from src.methods import load_and_format_data
    return load_and_format_data, os


@app.cell
def _(Path, os):
    project_name  = "2025-07-03_kpms-v2"
    model_name    = "2025-07-07_model-2"
    kpms_dir      = Path(os.environ["UNSUPERVISED_AGING"] + "/data/kpms_projects")
    poses_csv_dir = Path(os.environ["UNSUPERVISED_AGING"] + "/data/datasets/nature-aging_370/poses_csv")

    project_dir = kpms_dir / project_name
    return model_name, poses_csv_dir, project_dir


@app.cell
def _(kpms, model_name, project_dir):
    results = kpms.load_results(project_dir, model_name)
    return (results,)


@app.cell
def _(mo, np, results):
    _sequences = [_pose_dict["syllable"] for _pose_dict in results.values()]

    _unique_syllables = sorted({_s for _seq in _sequences for _s in _seq})
    _idx = {_s: _i for _i, _s in enumerate(_unique_syllables)}
    n = len(_unique_syllables)

    _T_counts = np.zeros((n, n), dtype=float)
    _U_counts = np.zeros(n, dtype=float)

    for _seq in mo.status.progress_bar(_sequences):
        for _s in _seq:
            _U_counts[_idx[_s]] += 1
        for _a, _b in zip(_seq[:-1], _seq[1:]):
            _T_counts[_idx[_a], _idx[_b]] += 1

    _row_sums = _T_counts.sum(axis=1, keepdims=True)
    T = _T_counts / _row_sums
    U = _U_counts / _U_counts.sum()

    print("n =", n)
    print("T shape:", T.shape)
    print("U shape:", U.shape)
    return T, U, n


@app.cell
def _(kpms, load_and_format_data, poses_csv_dir, project_dir):
    config_fn = lambda: kpms.load_config(project_dir)
    data, metadata, coordinates = load_and_format_data(poses_csv_dir, project_dir)
    return config_fn, coordinates


@app.cell
def _(Callable, Dict, List, Set, Tuple, Union, default_cost, np):
    CostFunc = Callable[[float, float, float, float], float]

    def hierarchical_motif_tree(
        n: int,                        
        U: Union[np.ndarray, list, tuple],
        T: Union[np.ndarray, list, tuple],
        cost_func: CostFunc = None
    ) -> np.ndarray:
        cost_func = cost_func or default_cost

        U = np.asarray(U, dtype=float)
        T = np.asarray(T, dtype=float)
        if U.shape != (n,) or T.shape != (n, n):
            raise ValueError("Shapes do not match 'n'")

        clusters: Dict[int, Set[int]] = {i: {i} for i in range(n)}
        probs: Dict[int, float] = {i: U[i] for i in range(n)}

        def trans_sum(A: Set[int], B: Set[int]) -> float:
            return T[np.ix_(list(A), list(B))].sum()

        linkage_rows: List[Tuple[float, float, float, int]] = []

        next_id = n

        while len(clusters) > 1:
            ids = list(clusters.keys())
            best_pair = None
            best_cost = np.inf

            for i_idx in range(len(ids) - 1):
                i = ids[i_idx]
                A = clusters[i]
                u_i = probs[i]
                for j_idx in range(i_idx + 1, len(ids)):
                    j = ids[j_idx]
                    B = clusters[j]
                    u_j = probs[j]

                    t_ij = trans_sum(A, B)
                    t_ji = trans_sum(B, A)

                    c = cost_func(u_i, u_j, t_ij, t_ji)
                    if c < best_cost:
                        best_cost = c
                        best_pair = (i, j)

            i, j = best_pair
            A, B = clusters.pop(i), clusters.pop(j)
            new_cluster = A | B
            clusters[next_id] = new_cluster
            probs[next_id] = probs.pop(i) + probs.pop(j)

            linkage_rows.append((i, j, best_cost, len(new_cluster)))
            next_id += 1

        return np.asarray(linkage_rows, dtype=float)
    return CostFunc, hierarchical_motif_tree


@app.cell
def _(CostFunc):
    _eps = 1e-12
    vame_cost_fn: CostFunc = lambda u_i, u_j, t_ij, t_ji: (u_i + u_j) / (t_ij + t_ji + _eps)
    alzh_cost_fn: CostFunc = lambda u_i, u_j, t_ij, t_ji: u_i + u_j / (t_ij + t_ji + _eps)
    return (alzh_cost_fn,)


@app.cell
def _():
    # kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config_fn())
    return


@app.cell
def _(coordinates):
    trans_coordinates = {k: v[..., ::-1] for k, v in coordinates.items()}
    return (trans_coordinates,)


@app.cell
def _(config_fn, kpms, model_name, project_dir, results, trans_coordinates):
    kpms.generate_grid_movies(results, project_dir, model_name, coordinates=trans_coordinates, **config_fn(), overlay_keypoints=True)
    print(f"rsync -avz miaod@login.sumner2.jax.org:{project_dir / model_name}/grid_movies Downloads/2025-07-09_grid_movies/")
    return


@app.cell
def _():
    # mo.md("trajectory-based syllable dendrogram")
    # kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config_fn())
    return


@app.cell
def _(
    T,
    U,
    alzh_cost_fn: "CostFunc",
    dendrogram,
    hierarchical_motif_tree,
    n,
    plt,
):
    linkage = hierarchical_motif_tree(n, U, T, alzh_cost_fn)

    fig, ax = plt.subplots(figsize=(64, 24))
    dendrogram(
        linkage,
        ax=ax,
        labels=[f"m{i}" for i in range(n)],
        distance_sort="ascending",
        color_threshold=None
    )
    ax.set_xlabel("leaf index")
    ax.set_ylabel("merge cost")
    ax.set_yscale("symlog")
    fig.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
