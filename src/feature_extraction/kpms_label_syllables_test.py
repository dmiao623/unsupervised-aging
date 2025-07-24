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

    from pathlib import Path

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    return Path, os


@app.cell
def _(os):
    import sys

    sys.path.append(os.environ["UNSUPERVISED_AGING"] + "/src/kpms_utils")

    from src.methods import load_and_format_data
    return (load_and_format_data,)


@app.cell
def _():
    import keypoint_moseq as kpms
    return (kpms,)


@app.cell
def _(Path, os):
    project_name  = "2025-07-16_kpms-v3"
    model_name    = "2025-07-16_model-4"
    kpms_dir      = Path(os.environ["UNSUPERVISED_AGING"]) / "data/kpms_projects"
    videos_dir    = Path(os.environ["UNSUPERVISED_AGING"]) / "data/datasets/geroscience_492/videos"
    poses_csv_dir = Path(os.environ["UNSUPERVISED_AGING"]) / "data/datasets/geroscience_492/poses_csv"

    project_dir = kpms_dir / project_name
    return model_name, poses_csv_dir, project_dir, videos_dir


@app.cell
def _(kpms, model_name, project_dir):
    moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
    moseq_df
    return


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
def _(config_fn, coordinates, kpms, model_name, project_dir, results):
    kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config_fn(),
                                   save_gifs=False, save_individually=False, get_limits_pctl=0.5)
    return


@app.cell
def _(
    config_fn,
    coordinates,
    kpms,
    model_name,
    project_dir,
    results,
    videos_dir,
):
    _tmp = config_fn()
    _tmp["video_dir"] = str(videos_dir)

    kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **_tmp, overlay_keypoints=True)
    return


@app.cell
def _(model_name, project_dir):
    from datetime import datetime

    _today = datetime.today().strftime('%Y-%m-%d')
    print(f"rsync -avz miaod@login.sumner2.jax.org:{project_dir / model_name}/grid_movies ~/Downloads/{_today}_kpms_grid_movies/")
    print(f"rsync -avz miaod@login.sumner2.jax.org:{project_dir / model_name}/trajectory_plots ~/Downloads/{_today}_kpms_grid_movies/")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
