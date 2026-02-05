"""Utility functions for Keypoint-MoSeq model training.

Provides the core model-fitting routine used by ``kpms_training.py``.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import jax
import keypoint_moseq as kpms
from src.utils import print_gpu_usage


def fit_and_save_model(
    model_name: str,
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    pca: Any,
    config_func: Callable[[], Dict[str, Any]],
    project_path: Path,
    *,
    full_model_iters: int,
    arhmm_iters: int,
    kappa: float,
    reduced_kappa: float,
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit a Keypoint-MoSeq model and save results to disk.

    Performs the full KPMS training pipeline:

    1. Initialize the model with the given PCA and config.
    2. Fit the AR-HMM with the *kappa* stickiness hyperparameter.
    3. Fit the full model with the *reduced_kappa* hyperparameter.
    4. Reindex syllables by frequency.
    5. Extract and save results as CSV.

    Args:
        model_name: Identifier for the trained model.
        data: Formatted pose data as returned by ``load_and_format_data``.
        metadata: Associated metadata dictionary.
        pca: Pre-computed PCA object (from ``kpms.io.load_pca``).
        config_func: Callable that returns the project configuration dict
            (typically ``lambda: kpms.load_config(project_dir)``).
        project_path: Path to the KPMS project directory.
        full_model_iters: Number of full model fitting iterations.
        arhmm_iters: Number of AR-HMM fitting iterations.
        kappa: Stickiness hyperparameter for AR-HMM fitting.
        reduced_kappa: Stickiness hyperparameter for full model fitting.
        seed: Random seed for model initialization.

    Returns:
        A tuple of ``(model, results)`` where *model* is the fitted KPMS
        model object and *results* is a dict of extracted results.
    """
    model = kpms.init_model(data, pca=pca, **config_func(), seed=seed)
    model = kpms.update_hypparams(model, kappa=kappa)

    print("\n--- FITTING AR-HMM ---\n")
    if jax.devices()[0].platform != "cpu":
        print_gpu_usage()

    model = kpms.fit_model(
        model,
        data,
        metadata,
        project_path,
        model_name,
        ar_only=True,
        num_iters=arhmm_iters,
        parallel_message_passing=False,
    )[0]

    print("\n--- FITTING FULL MODEL ---\n")
    if jax.devices()[0].platform != "cpu":
        print_gpu_usage()

    model = kpms.update_hypparams(model, kappa=reduced_kappa)
    model = kpms.fit_model(
        model,
        data,
        metadata,
        project_path,
        model_name,
        ar_only=False,
        start_iter=arhmm_iters,
        num_iters=arhmm_iters + full_model_iters,
        parallel_message_passing=False,
    )[0]

    print("\n--- REINDEXING SYLLABLES ---\n")
    kpms.reindex_syllables_in_checkpoint(project_path, model_name)

    print("\n--- SAVING RESULTS ---\n")
    results = kpms.extract_results(model, metadata, project_path, model_name)
    kpms.save_results_as_csv(results, project_path, model_name)

    return model, results
