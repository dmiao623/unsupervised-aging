import argparse
import json
import numpy as np
import pandas as pd

from functools import partial
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.neural_network import MLPRegressor
from typing import Sequence
from xgboost import XGBRegressor

from model_training_utils import TwoStageSearchCV, compute_nested_kfold_validation



def main(
    input_csv: str,
    xcat_json: str,
    output_path: str,
    seed: int,
    outer_n_splits: int,
    inner_n_splits: int,
    cpu_cores: int,
    X_cats: Sequence[str],
    y_cats: Sequence[str]
):

    regression_models = [
        partial(
            compute_nested_kfold_validation,
            estimator_ctor = lambda: Pipeline([
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]),
            param_grid = {},
            search_ctor = None,
            model_name = "Linear Regression",
        ),

        partial(
            compute_nested_kfold_validation,
            estimator_ctor = lambda: Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge(random_state=seed)),
            ]),
            param_grid = dict(
                model__alpha = np.logspace(-2, 3, 100)
            ),
            search_ctor = GridSearchCV,
            model_name = "Ridge Regression",
        ),

        partial(
            compute_nested_kfold_validation,
            estimator_ctor = lambda: Pipeline([
                ("scale", StandardScaler()),
                ("model", Lasso(max_iter=50000, random_state=seed)),
            ]),
            param_grid = dict(
                model__alpha = np.logspace(-2, 3, 100)
            ),
            search_ctor = GridSearchCV,
            model_name = "Lasso Regression",
        ),

        partial(
            compute_nested_kfold_validation,
            estimator_ctor = lambda: Pipeline([
                ("scale", StandardScaler()),
                ("model", ElasticNet(max_iter=50000, random_state=seed)),
            ]),
            param_grid = dict(
                model__alpha    = np.logspace(-2, 3, 100),
                model__l1_ratio = np.linspace(0.01, 0.99, 10)
            ),
            search_ctor = GridSearchCV,
            model_name = "Elastic Net",
        ),

        partial(
            compute_nested_kfold_validation,
            estimator_ctor = lambda: Pipeline([
                ("model", RandomForestRegressor(random_state=seed)) 
            ]),
            param_grid = dict(
                model__n_estimators           = np.linspace(100, 700, 5, dtype=int),
                model__max_depth              = [None, 10, 20],
                model__min_samples_split      = [2, 5, 10],
                model__min_samples_leaf       = [1, 2, 4],
                model__max_features           = ["sqrt", 0.8],
                model__min_impurity_decrease  = [0.0, 0.001],
            ),
            search_ctor = TwoStageSearchCV,
            search_kwargs = dict(
                first_param = "model__n_estimators",
                first_search_ctor = GridSearchCV,
                second_search_ctor = HalvingGridSearchCV,
            ),
            model_name = "Random Forest",
        ),

        partial(
            compute_nested_kfold_validation,
            estimator_ctor = lambda: Pipeline([
                ("model", XGBRegressor(random_state=seed, eval_metric="rmse"))
            ]),
            param_grid = dict(
                model__n_estimators      = np.linspace(200, 1200, 11, dtype=int),
                model__learning_rate     = [0.01, 0.03, 0.05, 0.10],
                model__max_depth         = [3, 4, 5, 6],
                model__subsample         = [0.60, 0.80, 1.00],
                model__colsample_bytree  = [0.60, 0.80, 1.00],
                model__min_child_weight  = [1, 3],
                model__gamma             = [0, 0.1],
                model__reg_alpha         = [0],
                model__reg_lambda        = [1],
            ),
            search_ctor = TwoStageSearchCV,
            search_kwargs = dict(
                first_param = "model__n_estimators",
                first_search_ctor = GridSearchCV,
                second_search_ctor = HalvingGridSearchCV,
            ),
            model_name = "XGBoost",
        ),

        partial(
            compute_nested_kfold_validation,
            estimator_ctor = lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    activation          = "relu",
                    solver              = "adam",
                    early_stopping      = True,
                    validation_fraction = 0.2,
                    max_iter            = 10000,
                    n_iter_no_change    = 10,
                    random_state        = seed,
                )),
            ]),
            param_grid = dict(
                model__hidden_layer_sizes = [
                    (64,),
                    (128,),
                    (128, 64),
                    (128, 64, 32),
                    (128, 64, 32, 32),
                    (128, 64, 32, 32, 32),
                ],
                model__alpha              = np.logspace(-5, -1, 5),
                model__learning_rate_init = [0.0003, 0.001, 0.005, 0.01],
                model__learning_rate      = ["constant", "adaptive"],
            ),
            search_ctor = TwoStageSearchCV,
            search_kwargs = dict(
                first_param = "model__hidden_layer_sizes",
                first_search_ctor = GridSearchCV,
                second_search_ctor = HalvingGridSearchCV,
            ),
            model_name = "Multi-Layer Perceptron",
        )
    ]

    data_df = pd.read_csv(input_csv)
    col_dict = json.load(open(xcat_json, "r"))
    groups = data_df["mouse_id"]

    all_runs = []
    for X_cat in X_cats:
        X = data_df[col_dict[X_cat]]
        for y_cat in y_cats:
            y = data_df[y_cat]

            print(f"\n(X_cat = {X_cat}, y_cat = {y_cat}):")
            for model in regression_models:
                df = model(
                    X = X,
                    y = y,
                    groups = groups,
                    outer_n_splits = outer_n_splits,
                    inner_n_splits = inner_n_splits,
                    n_jobs = cpu_cores,
                ).copy()

                df["X_cat"] = X_cat
                df["y_cat"] = y_cat
                df["model"] = model.keywords["model_name"]

                all_runs.append(df)

    results_df = pd.concat(all_runs, ignore_index=True)
    results_df.to_csv(output_path)
    print(f"Exported result dataframe to {output_path}.")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters for nested cross-validation")

    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--xcat_json", type=str, required=True,
                        help="Path to JSON file containing required X_cat column categories")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write output CSV file to")

    parser.add_argument("--seed", type=int, default=623,
                        help="Random seed value (default: 623)")
    parser.add_argument("--outer_n_splits", type=int, default=10,
                        help="Number of outer CV splits (default: 10)")
    parser.add_argument("--inner_n_splits", type=int, default=5,
                        help="Number of inner CV splits (default: 5)")
    parser.add_argument("--cpu_cores", type=int, default=1,
                        help="Number of CPU cores to use (default: 1)")
    parser.add_argument("--X_cats", nargs="+", default=[],
                        help="List of strings for X categorical features")
    parser.add_argument("--y_cats", nargs="+", default=["age", "fi"],
                        help="List of strings for y categorical features (either \"fi\" or \"age\"; default: both)")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---\n"
         f"input_csv      : {args.input_csv}\n"
         f"xcat_json      : {args.xcat_json}\n"
         f"output_path    : {args.output_path}\n"
         f"seed           : {args.seed}\n"
         f"outer_n_splits : {args.outer_n_splits}\n"
         f"inner_n_splits : {args.inner_n_splits}\n"
         f"cpu_cores      : {args.cpu_cores}\n"
         f"X_cats         : {args.X_cats}\n"
         f"y_cats         : {args.y_cats}\n"
         "-------------------"
    )

    main(args.input_csv, args.xcat_json, args.output_path, args.seed, args.outer_n_splits,
         args.inner_n_splits, args.cpu_cores, args.X_cats, args.y_cats)
