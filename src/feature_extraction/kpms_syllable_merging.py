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
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    return (os,)


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram

    from pathlib import Path
    from typing import Callable, Dict, Set, Tuple, List, Union

    import keypoint_moseq as kpms
    return Callable, Dict, List, Path, Set, Tuple, Union, kpms, mo, np, nx, plt


@app.cell
def _(os):
    import sys

    sys.path.append(os.environ["UNSUPERVISED_AGING"] + "/src/kpms_utils")

    from src.methods import load_and_format_data
    return (load_and_format_data,)


@app.cell
def _(Path, os):
    project_name  = "2025-07-03_kpms-v2"
    model_name    = "2025-07-07_model-2"
    kpms_dir      = Path(os.environ["UNSUPERVISED_AGING"]) / "data/kpms_projects"
    poses_csv_dir = Path(os.environ["UNSUPERVISED_AGING"]) / "data/datasets/nature-aging_634/poses_csv"

    project_dir = kpms_dir / project_name
    return model_name, poses_csv_dir, project_dir


@app.cell
def _(kpms, model_name, project_dir):
    results = kpms.load_results(project_dir, model_name)
    return (results,)


@app.cell
def _(kpms, load_and_format_data, poses_csv_dir, project_dir):
    config_fn = lambda: kpms.load_config(project_dir)
    _, _, coordinates = load_and_format_data(poses_csv_dir, project_dir)
    coordinates = {k: v[..., ::-1] for k, v in coordinates.items()}
    return config_fn, coordinates


@app.cell
def _(coordinates, results):
    _rm_key = []
    for key in results.keys():
        if key not in coordinates:
            _rm_key.append(key)
    for key in _rm_key:
        del results[key]
    len(results), len(coordinates) # should match
    return


@app.cell
def _(mo, np, results):
    def build_counts(results, th):
        sequences = [pose_dict["syllable"] for pose_dict in results.values()]
        unique_syllables = sorted({s for seq in sequences for s in seq})
        idx = {s: i for i, s in enumerate(unique_syllables)}
        n = len(unique_syllables)
        T_counts = np.zeros((n, n), dtype=float)
        U_counts = np.zeros(n, dtype=float)
        for seq in mo.status.progress_bar(sequences):
            for s in seq:
                U_counts[idx[s]] += 1
            for a, b in zip(seq[:-1], seq[1:]):
                T_counts[idx[a], idx[b]] += 1
        row_sums = T_counts.sum(axis=1, keepdims=True)
        T = np.divide(T_counts, row_sums, where=row_sums != 0)
        U = U_counts / U_counts.sum()
        keep = U >= th
        if not np.all(keep):
            unique_syllables = [s for s, k in zip(unique_syllables, keep) if k]
            idx = {s: i for i, s in enumerate(unique_syllables)}
            T = T[np.ix_(keep, keep)]
            U = U[keep]
            row_sums = T.sum(axis=1, keepdims=True)
            T = np.divide(T, row_sums, where=row_sums != 0)
            U /= U.sum()
            n = len(unique_syllables)
        print("n =", n)
        print("T shape:", T.shape)
        print("U shape:", U.shape)
        return n, T, U

    n, T, U = build_counts(results, th=0.)
    return T, U, n


@app.cell
def _(Callable, Dict, List, Set, T, Tuple, U, Union, default_cost, n, np):
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

    _eps = 1e-12
    vame_cost_fn: CostFunc = lambda u_i, u_j, t_ij, t_ji: (u_i + u_j) / (t_ij + t_ji + _eps)
    alzh_cost_fn: CostFunc = lambda u_i, u_j, t_ij, t_ji: u_i + u_j / (t_ij + t_ji + _eps)

    print(n, U.shape, T.shape)
    linkage = hierarchical_motif_tree(n, U, T, vame_cost_fn)
    return (linkage,)


@app.cell
def _():
    # kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config_fn())
    return


@app.cell
def _():
    # kpms.generate_grid_movies(results, project_dir, model_name, coordinates=trans_coordinates, **config_fn(), overlay_keypoints=True)
    # print(f"rsync -avz miaod@login.sumner2.jax.org:{project_dir / model_name}/grid_movies Downloads/2025-07-09_grid_movies/")
    return


@app.cell
def _(
    Path,
    config_fn,
    coordinates,
    kpms,
    mo,
    model_name,
    os,
    project_dir,
    results,
):
    _tmp = config_fn()
    _tmp["video_dir"] = Path(os.environ["UNSUPERVISED_AGING"]) / "data/datasets/nature-aging_634/videos"

    mo.md("trajectory-based syllable dendrogram")
    kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **_tmp)
    return


@app.cell
def _(linkage, nx, plt):
    import matplotlib as mpl
    mpl.style.use("default")

    def linkage_to_nx_tree(Z):
        n = Z.shape[0] + 1
        G = nx.DiGraph()
        for merge_idx, (c1, c2, *_rest) in enumerate(Z):
            parent = n + merge_idx
            G.add_edge(parent, int(c1))
            G.add_edge(parent, int(c2))
        return G, n

    def hierarchy_pos(G, root, leaf_spacing=2.5):
        pos = {}
        x_leaf = 0

        def _recurse(node, depth):
            nonlocal x_leaf
            kids = sorted(G.successors(node))
            if not kids:
                pos[node] = (x_leaf * leaf_spacing, -depth)
                x_leaf += 1
            else:
                for k in kids:
                    _recurse(k, depth + 1)
                xs = [pos[k][0] for k in kids]
                pos[node] = (sum(xs) / len(xs), -depth)

        _recurse(root, 0)
        return pos

    def plot_linkage_tree(
        Z,
        leaf_prefix="m",
        figsize=(16, 10),
        node_size=400,
        font_size=8,
        leaf_spacing=2.5,
    ):
        G, n_leaves = linkage_to_nx_tree(Z)
        root = max(G.nodes)
        pos = hierarchy_pos(G, root, leaf_spacing)

        labels = {i: f"{leaf_prefix}{i}" if i < n_leaves else "" for i in G}

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        nx.draw(
            G,
            pos,
            node_size=node_size,
            node_color="#1f77b4",
            edgecolors="black",
            linewidths=0.8,
            arrows=False,
            with_labels=False,
            ax=ax,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            font_size=font_size,
            verticalalignment="bottom",
            ax=ax,
        )

        ax.margins(0.05)
        ax.axis("off")
        fig.tight_layout()
        plt.show()

    plot_linkage_tree(linkage, leaf_spacing=3.5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
