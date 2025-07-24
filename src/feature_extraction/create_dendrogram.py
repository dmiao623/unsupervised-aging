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
    import sys

    from pathlib import Path

    import keypoint_moseq as kpms
    return Path, kpms, os, sys


@app.cell
def _():
    from keypoint_moseq.util import syllable_similarity
    return (syllable_similarity,)


@app.cell
def _(os, sys):
    sys.path.append(os.environ["UNSUPERVISED_AGING"] + "/src/kpms_utils")
    from src.methods import load_and_format_data
    return (load_and_format_data,)


@app.cell
def _(Path, os):
    unsupervised_aging_dir = Path(os.environ["UNSUPERVISED_AGING"])

    project_name  = "2025-07-03_kpms-v2"
    model_name    = "2025-07-07_model-2"
    kpms_dir      = unsupervised_aging_dir / "data/kpms_projects"
    dataset_dir   = unsupervised_aging_dir / "data/datasets/nature-aging_634/"
    poses_csv_dir = dataset_dir / "poses_csv"

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
    return (coordinates,)


@app.cell
def _(coordinates, linkage, results, squareform, syllable_similarity):
    distances, syllable_ixs = syllable_similarity(coordinates, results)
    Z = linkage(squareform(distances), "complete")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
