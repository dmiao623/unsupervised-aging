import numpy as np
import pandas as pd

from copy import deepcopy
from pathlib import Path
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from typing import Any, Callable, Mapping, MutableMapping, Sequence, Type, Union, Optional


class TwoStageSearchCV(BaseEstimator):
    """
    Hyper-parameter search that proceeds in **two sequential stages**.

    The search first tunes a primary hyper-parameter (``first_param``) using
    one cross-validation strategy/class (``first_search_ctor``).  
    After identifying the best value of that parameter, it freezes the
    parameter and performs a second search over the remaining parameters
    with a (possibly different) search class (``second_search_ctor``).  

    Parameters
    ----------
    estimator : BaseEstimator
        The base estimator (or pipeline) to be optimized.

    param_grid : Mapping[str, Sequence[Any]]
        Dictionary defining the full search space.  
        **Must** include ``first_param``.

    first_param : str
        Name of the hyper-parameter to be optimized in the first stage.

    first_search_ctor : Type[BaseSearchCV]
        Class used to perform the first-stage search
        (e.g. :class:`sklearn.model_selection.GridSearchCV`,
        :class:`sklearn.model_selection.HalvingGridSearchCV`, etc.).

    second_search_ctor : Type[BaseSearchCV]
        Class used for the second-stage search.

    cv : int or BaseCrossValidator
        Cross-validation splitter or number of folds used in **both** stages.

    scoring : str, default="neg_root_mean_squared_error"
        Scoring metric passed to both search objects.

    n_jobs : int, default=-1
        Number of parallel jobs for the underlying searches.

    refit : bool, default=True
        If ``True``, the final best estimator is refit on the full data after
        each stage; otherwise a fresh clone is refit only at the end.

    first_search_kwargs : dict, optional
        Extra keyword arguments forwarded to the first-stage search constructor.

    second_search_kwargs : dict, optional
        Extra keyword arguments forwarded to the second-stage search constructor.

    Attributes
    ----------
    best_estimator_ : BaseEstimator
        Estimator fitted on the full data with the best hyper-parameters found
        across both stages (available after :meth:`fit`).

    best_params_ : dict
        Mapping of all tuned hyper-parameters and their optimal values.

    cv_results_ : dict
        Cross-validation results of the **second** stage (or the first stage if
        no second-stage search was necessary).

    _stage1, _stage2 : BaseSearchCV
        Fitted search objects from stage 1 and stage 2 (the latter is present
        only if a second stage was executed).
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
        scoring: str                                             = "neg_root_mean_squared_error",
        n_jobs: int                                              = -1,
        refit: bool                                              = True,
        first_search_kwargs: Optional[MutableMapping[str, Any]]  = None,
        second_search_kwargs: Optional[MutableMapping[str, Any]] = None,
    ):
        self.estimator            = estimator
        self.param_grid           = deepcopy(param_grid)
        self.first_param          = first_param
        self.first_search_ctor    = first_search_ctor
        self.second_search_ctor   = second_search_ctor
        self.cv                   = cv
        self.scoring              = scoring
        self.n_jobs               = n_jobs
        self.refit                = refit
        self.first_search_kwargs  = first_search_kwargs or {}
        self.second_search_kwargs = second_search_kwargs or {}

        if first_param not in self.param_grid:
            raise ValueError(f"'{first_param}' not found in param_grid")

    def fit(self, X, y=None, **fit_params):
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

        stage2_grid = {k: v for k, v in self.param_grid.items() if k != self.first_param}
        if not stage2_grid:
            self.best_estimator_ = stage2_estimator.fit(X, y, **fit_params)
            self.best_params_    = {self.first_param: best_first_param}
            self.cv_results_     = self._stage1.cv_results_
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
            else clone(stage2_estimator).set_params(
                **{**{self.first_param: best_first_param},
                   **self._stage2.best_params_}
            ).fit(X, y, **fit_params)
        )
        self.best_params_ = {self.first_param: best_first_param,
                             **self._stage2.best_params_}
        self.cv_results_  = self._stage2.cv_results_
        return self

    def predict(self, X):
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
    scoring: str                                      = "neg_root_mean_squared_error",
    n_jobs: int                                       = -1,
    search_kwargs: Optional[MutableMapping[str, Any]] = None,
    model_name: str                                   = "",
    seed: int                                         = 623,
):
    """
    Perform nested cross-validation with group-aware splitting and hyperparameter tuning.

    Parameters
    ----------
    estimator_ctor : Callable[[], BaseEstimator]
        A callable that returns a new instance of the estimator (e.g., pipeline).
    
    param_grid : Mapping[str, Sequence[Any]]
        Dictionary of hyperparameters to search during inner cross-validation.
    
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    
    y : np.ndarray
        Target vector of shape (n_samples,).
    
    groups : np.ndarray
        Group labels for the samples, used for GroupKFold splitting.
    
    search_ctor : Type[BaseSearchCV] or None
        Class that performs the inner-loop hyper-parameter search
        (e.g. :class:`sklearn.model_selection.GridSearchCV`).
        If ``None``, the estimator is cloned and fitted once per outer fold
        with the parameters given in ``param_grid`` (which may be empty).
    
    outer_n_splits : int
        Number of splits for the outer cross-validation loop.
    
    inner_n_splits : int
        Number of splits for the inner cross-validation loop (for model selection).
    
    scoring : str, optional
        Scoring metric used for model evaluation during hyperparameter search.
        Default is "neg_root_mean_squared_error".
    
    n_jobs : int, optional
        Number of CPU cores to use for parallelization during hyperparameter search.
        Default is -1 (use all cores).
    
    search_kwargs : dict, optional
        Additional keyword arguments passed to the search constructor (e.g., TwoStageSearchCV).
    
    model_name : str, optional
        Optional name of the model, used for logging and progress tracking.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing predictions, ground truth, fold information, and best hyperparameters,
        for both training and test splits in each outer fold.
    """

    outer_cv = GroupKFold(n_splits=outer_n_splits, shuffle=True, random_state=seed)
    feature_names = (
        list(X.columns) if hasattr(X, "columns") else [f"feat_{i}" for i in range(X.shape[1])]
    )

    rows = []
    search_kwargs = search_kwargs or {}

    def _run_one_fold(fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray):
        sample_idx_train, sample_idx_test = train_idx, test_idx

        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test  = X.iloc[test_idx]  if hasattr(X, "iloc") else X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train    = groups[train_idx]

        if search_ctor is None:
            base_est = estimator_ctor()
            if param_grid:
                base_est.set_params(**{k: v[0] if isinstance(v, Sequence) else v
                                    for k, v in param_grid.items()})
            base_est.fit(X_train, y_train)
            best_model  = base_est
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

            best_model  = grid.best_estimator_
            best_params = grid.best_params_

        y_pred_train = best_model.predict(X_train)
        y_pred_test  = best_model.predict(X_test)

        def _make_df(X_block, y_block, y_pred_block, split_label, sample_idx_block):
            base = pd.DataFrame(X_block, columns=feature_names)
            base["y_true"]    = y_block
            base["y_pred"]    = y_pred_block
            base["fold"]      = fold_idx
            base["split"]     = split_label
            base["sample_idx"] = sample_idx_block
            for p_name, p_val in best_params.items():
                base[p_name] = str(p_val)
            return base

        return (
            _make_df(X_train, y_train, y_pred_train, "train", sample_idx_train),
            _make_df(X_test,  y_test,  y_pred_test,  "test",  sample_idx_test),
        )

    for fold, (tr_idx, te_idx) in enumerate(
        tqdm(outer_cv.split(X, y, groups=groups), total=outer_n_splits, desc=model_name)
    ):
        df_tr, df_te = _run_one_fold(fold, tr_idx, te_idx)
        rows.extend([df_tr, df_te])

    predictions_df = pd.concat(rows, axis=0, ignore_index=True)
    return predictions_df
