import marimo

__generated_with = "0.16.2"
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


if __name__ == "__main__":
    app.run()
