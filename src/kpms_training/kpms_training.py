"""Train a Keypoint-MoSeq model on preprocessed pose data.

Loads formatted pose data from a project directory, performs model fitting
using the Keypoint-MoSeq pipeline (AR-HMM then full model), and saves the
resulting model and extracted results to disk.

SLURM Template:
    scripts/templates/kpms_training.sh

Usage::

    python kpms_training.py \\
        --project_name <project_name> \\
        --model_name <model_name> \\
        --kpms_dir <path_to_kpms_projects> \\
        --videos_dir <path_to_videos> \\
        --poses_csv_dir <path_to_pose_csvs> \\
        [--g_mixed_map_iters <int>] \\
        [--g_arhmm_iters <int>] \\
        [--g_full_model_iters <int>] \\
        [--g_kappa <float>] \\
        [--g_reduced_kappa <float>] \\
        [--seed <int>]

Note:
    Assumes that PCA has already been performed for the project and saved
    using ``kpms.save_pca``. The project must be created using a standard
    KPMS set-up, e.g. via ``create_kpms_project.py``.
"""

import argparse
from pathlib import Path

import jax
import keypoint_moseq as kpms
from jax_moseq.utils import set_mixed_map_iters

from src.utils import set_up_logging, print_gpu_usage
from src.methods import load_and_format_data
from .kpms_training_utils import fit_and_save_model


def main(
    project_name: str,
    model_name: str,
    kpms_dir: str,
    videos_dir: str,
    poses_csv_dir: str,
    g_mixed_map_iters: int,
    g_arhmm_iters: int,
    g_full_model_iters: int,
    g_kappa: float,
    g_reduced_kappa: float,
    seed: int,
):
    """Train a single Keypoint-MoSeq model and save results.

    Loads pose data, loads a previously computed PCA, then fits an AR-HMM
    followed by the full KPMS model. Syllables are reindexed by frequency
    and the results are saved as CSV.

    Args:
        project_name: Name of the KPMS project (subdirectory of *kpms_dir*).
        model_name: Identifier for the trained model.
        kpms_dir: Parent directory containing KPMS project directories.
        videos_dir: Directory containing the raw video files.
        poses_csv_dir: Directory containing pose estimation results as CSVs.
        g_mixed_map_iters: Degree of computation serialization; reduce if
            running out of GPU memory.
        g_arhmm_iters: Number of AR-HMM fitting iterations.
        g_full_model_iters: Number of full model fitting iterations.
        g_kappa: Stickiness hyperparameter for AR-HMM fitting.
        g_reduced_kappa: Stickiness hyperparameter for full model fitting.
        seed: Random seed for model initialization.

    Raises:
        FileNotFoundError: If *kpms_dir*, *videos_dir*, or *poses_csv_dir*
            does not exist.
    """
    set_mixed_map_iters(g_mixed_map_iters)

    kpms_dir = Path(kpms_dir)
    videos_dir = Path(videos_dir)
    poses_csv_dir = Path(poses_csv_dir)
    project_dir = kpms_dir / project_name

    for d in (kpms_dir, videos_dir, poses_csv_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Directory '{d}' does not exist.")

    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    set_up_logging(log_dir)

    if jax.devices()[0].platform != "cpu":
        print_gpu_usage()

    print("\n--- DATA LOADING + FORMATTING ---\n")
    data, metadata, _ = load_and_format_data(str(poses_csv_dir), str(project_dir))

    print("\n--- PCA ANALYSIS ---\n")
    config_fn = lambda: kpms.load_config(str(project_dir))
    pca = kpms.io.load_pca(project_dir)

    if jax.devices()[0].platform != "cpu":
        print_gpu_usage()

    print("\n--- FITTING MODELS ---\n")
    fit_and_save_model(
        model_name,
        data,
        metadata,
        pca,
        config_fn,
        project_dir,
        full_model_iters=g_full_model_iters,
        arhmm_iters=g_arhmm_iters,
        kappa=g_kappa,
        reduced_kappa=g_reduced_kappa,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Keypoint-MoSeq model on preprocessed pose data.",
    )

    # Project configuration
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the keypoint-MoSeq project",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of keypoint-MoSeq model"
    )
    parser.add_argument(
        "--kpms_dir",
        type=str,
        required=True,
        help="Path of the keypoint-MoSeq project directory",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Path that contains the raw videos",
    )
    parser.add_argument(
        "--poses_csv_dir",
        type=str,
        required=True,
        help="Path that contains the pose data in CSV format",
    )

    # Model training parameters
    parser.add_argument(
        "--g_mixed_map_iters",
        type=int,
        default=8,
        help="Degree of serializing computations; reduce if running out of GPU memory (default: 8)",
    )
    parser.add_argument(
        "--g_arhmm_iters",
        type=int,
        default=100,
        help="Number of AR-HMM fitting iterations (default: 100)",
    )
    parser.add_argument(
        "--g_full_model_iters",
        type=int,
        default=400,
        help="Number of full model fitting iterations (default: 400)",
    )
    parser.add_argument(
        "--g_kappa",
        type=float,
        default=1e6,
        help="Stickiness hyperparameter for AR HMM fitting (default: 1e6)",
    )
    parser.add_argument(
        "--g_reduced_kappa",
        type=float,
        default=1e5,
        help="Stickiness hyperparameter for full model fitting (default: 1e5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=623,
        help="Seed for model fitting initialization (default: 623)",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(
        args.project_name,
        args.model_name,
        args.kpms_dir,
        args.videos_dir,
        args.poses_csv_dir,
        args.g_mixed_map_iters,
        args.g_arhmm_iters,
        args.g_full_model_iters,
        args.g_kappa,
        args.g_reduced_kappa,
        args.seed,
    )
