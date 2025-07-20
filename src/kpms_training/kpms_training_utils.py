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
) -> Tuple[Any, str, Dict[str, Any]]:

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
