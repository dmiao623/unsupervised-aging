import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import types, param
    if not hasattr(param, "reactive"):
        param.reactive = types.SimpleNamespace(rx=param.Parameterized)

    del types, param
    return


@app.cell
def _():


    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns
    import sys
    import umap

    from pathlib import Path
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    return Path, json, linkage, nx, os, pd, plt, squareform, sys


@app.cell
def _():
    import keypoint_moseq as kpms
    from keypoint_moseq.io import _get_path
    from keypoint_moseq.util import syllable_similarity
    return kpms, syllable_similarity


@app.cell
def _(os, sys):
    sys.path.append(os.environ["UNSUPERVISED_AGING"] + "/src/kpms_utils")
    from src.methods import load_and_format_data
    return (load_and_format_data,)


@app.cell
def _(Path, json, os, pd):
    unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])
    kpms_dir               = unsupervised_aging_dir / "data/kpms_projects"
    data_info_path         = unsupervised_aging_dir / "data/data_info.json"

    with data_info_path.open("r") as f:
        data_info = json.load(f)

    data = {}
    for dataset, info in data_info.items():
        with Path(info["xcats_path"]).open("r") as f:
            xcats = json.load(f)
        data[dataset] = {
            "features": pd.read_csv(info["features_path"], index_col=0),
            "xcats": xcats,
            "results": pd.read_csv(info["model_eval_path"], index_col=0, low_memory=False)
        }
    data_info
    return data_info, kpms_dir


@app.cell
def _(Path, data_info, kpms, kpms_dir):
    dataset_id = "combined_1126"
    project_name_suffix = "_150"

    project_name, model_name = data_info[dataset_id]["kpms_project"], data_info[dataset_id]["kpms_model"]
    poses_csv_dir = Path(data_info[dataset_id]["dataset_dir"]) / "poses_csv"
    project_dir = kpms_dir / (project_name + project_name_suffix)

    results = kpms.load_results(project_dir, model_name)
    return model_name, poses_csv_dir, project_dir, results


@app.cell
def _(kpms, load_and_format_data, poses_csv_dir, project_dir):
    config_fn = lambda: kpms.load_config(project_dir)
    _, _, coordinates = load_and_format_data(poses_csv_dir, project_dir)
    coordinates = {k: v[..., ::-1] for k, v in coordinates.items()}
    return config_fn, coordinates


@app.cell
def _(
    Path,
    config_fn,
    coordinates,
    linkage,
    model_name,
    nx,
    os,
    plt,
    project_dir,
    results,
    squareform,
    syllable_similarity,
):
    def plot_similarity_dendrogram_nx(
        coordinates,
        results,
        project_dir=None,
        model_name=None,
        save_path=None,
        metric="cosine",
        pre=0.167,
        post=0.5,
        min_frequency=0.005,
        min_duration=3,
        bodyparts=None,
        use_bodyparts=None,
        density_sample=False,
        sampling_options={"n_neighbors": 50},
        figsize=(8, 5),
        fps=None,
        **kwargs,
    ):
        assert fps is not None, "fps must be provided"
        pre = round(pre * fps)
        post = round(post * fps)

        save_path = _get_path(project_dir, model_name, save_path, "similarity_dendrogram_nx")

        distances, syllable_ixs = syllable_similarity(
            coordinates,
            results,
            metric,
            pre,
            post,
            min_frequency,
            min_duration,
            bodyparts,
            use_bodyparts,
            density_sample,
            sampling_options,
        )

        Z = linkage(squareform(distances), method="complete")

        G = nx.DiGraph()
        n_leaves = len(syllable_ixs)
        labels = {i: f"Syllable {s}" for i, s in enumerate(syllable_ixs)}
        node_heights = {}

        for i, (a, b, dist, _) in enumerate(Z):
            node_id = i + n_leaves
            G.add_edge(node_id, int(a))
            G.add_edge(node_id, int(b))
            node_heights[node_id] = dist

        pos = _hierarchy_pos(G, max(G.nodes), width=1.5)

        plt.figure(figsize=figsize)
        nx.draw(
            G,
            pos=pos,
            labels={k: labels.get(k, "") for k in G.nodes},
            with_labels=True,
            node_size=50,
            font_size=8,
            font_color="black",
            edge_color="gray",
        )
        plt.title("Syllable similarity dendrogram (NetworkX)")
        plt.axis("off")

        print(f"Saving dendrogram plot to {save_path}")
        for ext in ["pdf", "png"]:
            plt.savefig(save_path + "." + ext)
        plt.close()


    def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G, child, width=dx, vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root
                )
        return pos

    _tmp = config_fn()
    _tmp["video_dir"] = Path(os.environ["UNSUPERVISED_AGING"]) / "data/datasets/nature-aging_634/videos"
    _tmp["fps"] = 30
    plot_similarity_dendrogram_nx(coordinates, results, project_dir, model_name, **_tmp)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
