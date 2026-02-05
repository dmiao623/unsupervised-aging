"""Run nested cross-validation for regression models on input features.

Loads a CSV file containing feature and target variables along with a JSON
mapping of feature groupings. Performs nested group-aware cross-validation
using several regression models (Elastic Net, Random Forest, XGBoost,
Multi-Layer Perceptron), each with appropriate hyperparameter tuning.
Model predictions, parameters, and performance metrics are aggregated and
exported to CSV.

SLURM Template:
    scripts/templates/model_evaluation.sh

Usage::

    python model_evaluation.py \\
        --input_csv <path_to_input_csv> \\
        --xcat_json <path_to_xcat_json> \\
        --output_path <path_to_output_csv> \\
        [--seed <int>] \\
        [--n_repeats <int>] \\
        [--outer_n_splits <int>] \\
        [--inner_n_splits <int>] \\
        [--cpu_cores <int>] \\
        [--X_cats <list_of_feature_groups>] \\
        [--y_cats <list_of_target_columns>]

Note:
    The input CSV must include a column named ``mouse_id`` used for grouped
    splitting. The JSON file should map X category names (strings) to lists
    of column names in the CSV. Target columns specified in ``--y_cats``
    must exist in the CSV.
"""

import argparse
import json
import numpy as np
import pandas as pd

from functools import partial
from pathlib import Path
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from typing import Sequence
from xgboost import XGBRegressor

from model_evaluation_utils import TwoStageSearchCV, compute_nested_kfold_validation


def main(
    input_csv: str,
    xcat_json: str,
    output_path: str,
    seed: int,
    n_repeats: int,
    outer_n_splits: int,
    inner_n_splits: int,
    cpu_cores: int,
    X_cats: Sequence[str],
    y_cats: Sequence[str],
    export_all: bool,
    export_individual: bool,
):
    """Train and evaluate regression models via nested cross-validation.

    For every combination of feature group (*X_cats*) and target column
    (*y_cats*), each regression model is trained inside a nested
    group-aware cross-validation loop. Results (predictions, fold info,
    best hyperparameters) are collected and written to CSV.

    Args:
        input_csv: Path to the input CSV file with features and targets.
        xcat_json: Path to a JSON file mapping feature-group names to
            lists of column names.
        output_path: Path for the combined output CSV file.
        seed: Base random seed.
        n_repeats: Number of times to repeat cross-validation per model.
        outer_n_splits: Number of outer CV folds.
        inner_n_splits: Number of inner CV folds (hyperparameter search).
        cpu_cores: Number of CPU cores for parallel search.
        X_cats: Feature group names (keys in *xcat_json*).
        y_cats: Target column names (e.g. ``["age", "fi"]``).
        export_all: If ``True``, export a single combined CSV.
        export_individual: If ``True``, export per-group CSV files.

    Raises:
        AssertionError: If neither *export_all* nor *export_individual*
            is set.
    """
    assert export_all or export_individual, "Nothing to export!"
    output_path = Path(output_path)

    regression_models = [
        partial(
            compute_nested_kfold_validation,
            estimator_ctor=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(max_iter=50000, random_state=seed)),
                ]
            ),
            param_grid=dict(
                model__alpha=np.logspace(-2, 3, 100),
                model__l1_ratio=np.linspace(0.01, 0.99, 10),
            ),
            search_ctor=GridSearchCV,
            model_name="Elastic Net",
        ),
        partial(
            compute_nested_kfold_validation,
            estimator_ctor=lambda: Pipeline(
                [
                    ("model", RandomForestRegressor(random_state=seed)),
                ]
            ),
            param_grid=dict(
                model__n_estimators=np.linspace(100, 700, 5, dtype=int),
                model__max_depth=[None, 10, 20],
                model__min_samples_split=[2, 5, 10],
                model__min_samples_leaf=[1, 2, 4],
                model__max_features=["sqrt", 0.8],
                model__min_impurity_decrease=[0.0, 0.001],
            ),
            search_ctor=TwoStageSearchCV,
            search_kwargs=dict(
                first_param="model__n_estimators",
                first_search_ctor=GridSearchCV,
                second_search_ctor=HalvingGridSearchCV,
            ),
            model_name="Random Forest",
        ),
        partial(
            compute_nested_kfold_validation,
            estimator_ctor=lambda: Pipeline(
                [
                    ("model", XGBRegressor(random_state=seed, eval_metric="rmse")),
                ]
            ),
            param_grid=dict(
                model__n_estimators=np.linspace(200, 1200, 11, dtype=int),
                model__learning_rate=[0.01, 0.03, 0.05, 0.10],
                model__max_depth=[3, 4, 5, 6],
                model__subsample=[0.60, 0.80, 1.00],
                model__colsample_bytree=[0.60, 0.80, 1.00],
                model__min_child_weight=[1, 3],
                model__gamma=[0, 0.1],
                model__reg_alpha=[0],
                model__reg_lambda=[1],
            ),
            search_ctor=TwoStageSearchCV,
            search_kwargs=dict(
                first_param="model__n_estimators",
                first_search_ctor=GridSearchCV,
                second_search_ctor=HalvingGridSearchCV,
            ),
            model_name="XGBoost",
        ),
        partial(
            compute_nested_kfold_validation,
            estimator_ctor=lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            activation="relu",
                            solver="adam",
                            early_stopping=True,
                            validation_fraction=0.2,
                            max_iter=100000,
                            n_iter_no_change=10,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            param_grid=dict(
                model__hidden_layer_sizes=[
                    (64,),
                    (128,),
                    (128, 64),
                    (128, 64, 32),
                    (128, 64, 32, 32),
                    (128, 64, 32, 32, 32),
                ],
                model__alpha=np.logspace(-5, -1, 5),
                model__learning_rate_init=[0.0003, 0.001, 0.005, 0.01],
                model__learning_rate=["constant", "adaptive"],
            ),
            search_ctor=TwoStageSearchCV,
            search_kwargs=dict(
                first_param="model__hidden_layer_sizes",
                first_search_ctor=GridSearchCV,
                second_search_ctor=HalvingGridSearchCV,
            ),
            model_name="Multi-Layer Perceptron",
        ),
    ]

    data_df = pd.read_csv(input_csv)
    with open(xcat_json, "r") as f:
        col_dict = json.load(f)
    groups = data_df["mouse_id"]

    all_runs = []
    for X_cat in X_cats:
        X = data_df[col_dict[X_cat]]
        for y_cat in y_cats:
            y = data_df[y_cat]

            print(f"\n(X_cat = {X_cat}, y_cat = {y_cat}):")
            dfs = []
            for model in regression_models:
                for repeat in range(n_repeats):
                    temp_df = model(
                        X=X,
                        y=y,
                        groups=groups,
                        outer_n_splits=outer_n_splits,
                        inner_n_splits=inner_n_splits,
                        n_jobs=cpu_cores,
                        seed=seed + repeat,
                    )
                    temp_df["repeat"] = repeat
                    temp_df["model"] = model.keywords["model_name"]
                    dfs.append(temp_df)

            df = pd.concat(dfs, ignore_index=True)
            df["X_cat"] = X_cat
            df["y_cat"] = y_cat
            all_runs.append(df)

            if export_individual:
                temp_path = (
                    output_path.parent
                    / f"{output_path.stem}__{X_cat}__{y_cat}{output_path.suffix}"
                )
                df.to_csv(temp_path)
                print(f"Exported individual dataframe to {temp_path}.")

    if export_all:
        results_df = pd.concat(all_runs, ignore_index=True)
        results_df.to_csv(output_path)
        print(f"Exported result dataframe to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run nested cross-validation for regression models.",
    )

    # I/O paths
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--xcat_json",
        type=str,
        required=True,
        help="Path to JSON file containing required X_cat column categories",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write output CSV file to",
    )

    # Cross-validation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=623,
        help="Random seed value (default: 623)",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="Number of times to run cross-validation (default: 1)",
    )
    parser.add_argument(
        "--outer_n_splits",
        type=int,
        default=10,
        help="Number of outer CV splits (default: 10)",
    )
    parser.add_argument(
        "--inner_n_splits",
        type=int,
        default=5,
        help="Number of inner CV splits (default: 5)",
    )
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=1,
        help="Number of CPU cores to use (default: 1)",
    )

    # Feature / target selection
    parser.add_argument(
        "--X_cats",
        nargs="+",
        default=[],
        help="List of feature group names (keys in xcat_json)",
    )
    parser.add_argument(
        "--y_cats",
        nargs="+",
        default=["age", "fi"],
        help="Target column names (default: age fi)",
    )

    # Export options
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="Export combined results as a single CSV",
    )
    parser.add_argument(
        "--export_individual",
        action="store_true",
        help="Export each (X_cat, y_cat) result as a separate CSV",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(
        args.input_csv,
        args.xcat_json,
        args.output_path,
        args.seed,
        args.n_repeats,
        args.outer_n_splits,
        args.inner_n_splits,
        args.cpu_cores,
        args.X_cats,
        args.y_cats,
        args.export_all,
        args.export_individual,
    )
