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
    import keypoint_moseq as kpms
    return (kpms,)


@app.cell
def _():
    import numpy as np
    import os

    from pathlib import Path
    return Path, np, os


@app.cell
def _(Path, os):
    project_dir       = Path(os.environ["UNSUPERVISED_AGING"])
    kpms_dir          = project_dir / "data/kpms_projects"
    kpms_project_name = "2025-07-03_kpms-v2"
    kpms_project_dir  = kpms_dir / kpms_project_name

    num_models        = 8                    # number of models to compare
    model_basename    = "2025-07-07_model-"  # models names (`{model_basename}i` for i ∈ [1, num_models])

    model_names = [f"{model_basename}{i}" for i in range(1, num_models+1)]
    return kpms_project_dir, model_names


@app.cell
def _(kpms, kpms_project_dir, model_names):
    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(kpms_project_dir, model_names)
    return eml_scores, eml_std_errs


@app.cell
def _(eml_scores, model_names, np):
    best_model_name = model_names[np.argmax(eml_scores)]
    print(f"Best model: {best_model_name}")
    return


@app.cell
def _(eml_scores, eml_std_errs, kpms, model_names):
    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
