"""Utility classes and functions for nested cross-validation model evaluation.

Provides ``TwoStageSearchCV``, a two-stage hyperparameter search estimator,
and ``compute_nested_kfold_validation``, which performs nested group-aware
cross-validation. Used by ``model_evaluation.py``.
"""

import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
    Type,
    Union,
    Optional,
)


class TwoStageSearchCV(BaseEstimator):
    """Hyperparameter search that proceeds in two sequential stages.

    First tunes a primary hyperparameter (*first_param*) using one
    cross-validation search class (*first_search_ctor*). After identifying
    the best value, it freezes that parameter and performs a second search
    over the remaining parameters with a (possibly different) search class
    (*second_search_ctor*).

    Args:
        estimator: The base estimator (or pipeline) to be optimized.
        param_grid: Dictionary defining the full search space. Must
            include *first_param*.
        first_param: Name of the hyperparameter optimized in the first
            stage.
        first_search_ctor: Class used for the first-stage search (e.g.
            ``GridSearchCV``).
        second_search_ctor: Class used for the second-stage search.
        cv: Cross-validation splitter or number of folds used in both
            stages.
        scoring: Scoring metric passed to both search objects.
        n_jobs: Number of parallel jobs for the underlying searches.
        refit: If ``True``, refit the best estimator on the full data
            after each stage.
        first_search_kwargs: Extra keyword arguments forwarded to the
            first-stage search constructor.
        second_search_kwargs: Extra keyword arguments forwarded to the
            second-stage search constructor.

    Attributes:
        best_estimator\_: Estimator fitted with the best hyperparameters
            found across both stages (available after ``fit``).
        best_params\_: Mapping of all tuned hyperparameters and their
            optimal values.
        cv_results\_: Cross-validation results of the second stage (or
            the first stage if no second-stage search was necessary).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Mapping[str, Sequence[Any]],
        *,
        first_param: str,
        first_search_ctor: Type[BaseSearchCV],
        second_search_ctor: Type[BaseSearchCV],
        cv: Union[int, BaseCrossValidator],
        scoring: str = "neg_root_mean_squared_error",
        n_jobs: int = -1,
        refit: bool = True,
        first_search_kwargs: Optional[MutableMapping[str, Any]] = None,
        second_search_kwargs: Optional[MutableMapping[str, Any]] = None,
    ):
        self.estimator = estimator
        self.param_grid = deepcopy(param_grid)
        self.first_param = first_param
        self.first_search_ctor = first_search_ctor
        self.second_search_ctor = second_search_ctor
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.first_search_kwargs = first_search_kwargs or {}
        self.second_search_kwargs = second_search_kwargs or {}

        if first_param not in self.param_grid:
            raise ValueError(f"'{first_param}' not found in param_grid")

    def fit(self, X, y=None, **fit_params):
        """Run the two-stage hyperparameter search.

        Args:
            X: Training feature matrix.
            y: Target values.
            **fit_params: Additional parameters passed to the estimator's
                ``fit`` method.

        Returns:
            self
        """
        stage1_grid = {self.first_param: self.param_grid[self.first_param]}
        self._stage1 = self.first_search_ctor(
            clone(self.estimator),
            param_grid=stage1_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            **self.first_search_kwargs,
        )

        self._stage1.fit(X, y, **fit_params)
        best_first_param = self._stage1.best_params_[self.first_param]

        stage2_estimator = clone(self.estimator)
        stage2_estimator.set_params(**{self.first_param: best_first_param})

        stage2_grid = {
            k: v for k, v in self.param_grid.items() if k != self.first_param
        }
        if not stage2_grid:
            self.best_estimator_ = stage2_estimator.fit(X, y, **fit_params)
            self.best_params_ = {self.first_param: best_first_param}
            self.cv_results_ = self._stage1.cv_results_
            return self

        self._stage2 = self.second_search_ctor(
            stage2_estimator,
            param_grid=stage2_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            **self.second_search_kwargs,
        )

        self._stage2.fit(X, y, **fit_params)

        self.best_estimator_ = (
            self._stage2.best_estimator_
            if self.refit
            else clone(stage2_estimator)
            .set_params(
                **{**{self.first_param: best_first_param}, **self._stage2.best_params_}
            )
            .fit(X, y, **fit_params)
        )
        self.best_params_ = {
            self.first_param: best_first_param,
            **self._stage2.best_params_,
        }
        self.cv_results_ = self._stage2.cv_results_
        return self

    def predict(self, X):
        """Predict using the best estimator found during ``fit``.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)


def compute_nested_kfold_validation(
    estimator_ctor: Callable[[], BaseEstimator],
    param_grid: Mapping[str, Sequence[Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    search_ctor: Optional[Type[BaseSearchCV]],
    outer_n_splits: int,
    inner_n_splits: int,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    search_kwargs: Optional[MutableMapping[str, Any]] = None,
    model_name: str = "",
    seed: int = 623,
) -> pd.DataFrame:
    """Perform nested cross-validation with group-aware splitting.

    The outer loop uses ``GroupKFold`` to partition samples by group. In
    each outer fold, an inner hyperparameter search (using *search_ctor*)
    selects the best model configuration, which is then evaluated on the
    held-out test set.

    Args:
        estimator_ctor: Callable that returns a new estimator instance.
        param_grid: Hyperparameter search space for the inner CV loop.
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Target vector of shape ``(n_samples,)``.
        groups: Group labels for ``GroupKFold`` splitting.
        search_ctor: Class for inner-loop hyperparameter search (e.g.
            ``GridSearchCV``). If ``None``, the estimator is fitted
            directly with the parameters in *param_grid*.
        outer_n_splits: Number of outer CV folds.
        inner_n_splits: Number of inner CV folds.
        scoring: Scoring metric for hyperparameter search.
        n_jobs: Number of parallel jobs for the search.
        search_kwargs: Extra keyword arguments passed to *search_ctor*.
        model_name: Label for progress-bar display.
        seed: Random seed for outer fold shuffling.

    Returns:
        A ``DataFrame`` containing predictions, ground truth, fold
        information, and best hyperparameters for both training and test
        splits in each outer fold.
    """
    outer_cv = GroupKFold(n_splits=outer_n_splits, shuffle=True, random_state=seed)
    feature_names = (
        list(X.columns)
        if hasattr(X, "columns")
        else [f"feat_{i}" for i in range(X.shape[1])]
    )

    rows = []
    search_kwargs = search_kwargs or {}

    def _run_one_fold(fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray):
        """Fit one outer fold and return train/test DataFrames."""
        sample_idx_train, sample_idx_test = train_idx, test_idx

        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        if search_ctor is None:
            base_est = estimator_ctor()
            if param_grid:
                base_est.set_params(
                    **{
                        k: v[0] if isinstance(v, Sequence) else v
                        for k, v in param_grid.items()
                    }
                )
            base_est.fit(X_train, y_train)
            best_model = base_est
            best_params = {}
        else:
            grid = search_ctor(
                estimator_ctor(),
                param_grid=param_grid,
                cv=GroupKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed),
                scoring=scoring,
                n_jobs=n_jobs,
                **search_kwargs,
            )
            grid.fit(X_train, y_train, groups=groups_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        def _make_df(X_block, y_block, y_pred_block, split_label, sample_idx_block):
            """Build a results DataFrame for one split of one fold."""
            base = pd.DataFrame(X_block, columns=feature_names)
            base["y_true"] = y_block
            base["y_pred"] = y_pred_block
            base["fold"] = fold_idx
            base["split"] = split_label
            base["sample_idx"] = sample_idx_block
            for p_name, p_val in best_params.items():
                base[p_name] = str(p_val)
            return base

        return (
            _make_df(X_train, y_train, y_pred_train, "train", sample_idx_train),
            _make_df(X_test, y_test, y_pred_test, "test", sample_idx_test),
        )

    for fold, (tr_idx, te_idx) in enumerate(
        tqdm(outer_cv.split(X, y, groups=groups), total=outer_n_splits, desc=model_name)
    ):
        df_tr, df_te = _run_one_fold(fold, tr_idx, te_idx)
        rows.extend([df_tr, df_te])

    predictions_df = pd.concat(rows, axis=0, ignore_index=True)
    return predictions_df
