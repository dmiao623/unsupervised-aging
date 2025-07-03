import argparse
import logging
import os
import sys

from pathlib import Path

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import keypoint_moseq as kpms
from jax_moseq.utils import set_mixed_map_iters

from src.utils import set_up_logging, print_gpu_usage, validate_data_quality
from src.methods import load_and_format_data, perform_pca
from kpms_training_utils import fit_and_save_model


G_BODYPARTS = [
    "NOSE_INDEX",
    "LEFT_EAR_INDEX",
    "RIGHT_EAR_INDEX",
    "BASE_NECK_INDEX",
    "LEFT_FRONT_PAW_INDEX",
    "RIGHT_FRONT_PAW_INDEX",
    "CENTER_SPINE_INDEX",
    "LEFT_REAR_PAW_INDEX",
    "RIGHT_REAR_PAW_INDEX",
    "BASE_TAIL_INDEX",
    "MID_TAIL_INDEX",
    "TIP_TAIL_INDEX",
]

G_SKELETON = [
    ["TIP_TAIL_INDEX", "MID_TAIL_INDEX"],
    ["MID_TAIL_INDEX", "BASE_TAIL_INDEX"],
    ["BASE_TAIL_INDEX", "RIGHT_REAR_PAW_INDEX"],
    ["BASE_TAIL_INDEX", "LEFT_REAR_PAW_INDEX"],
    ["BASE_TAIL_INDEX", "CENTER_SPINE_INDEX"],
    ["CENTER_SPINE_INDEX", "LEFT_FRONT_PAW_INDEX"],
    ["CENTER_SPINE_INDEX", "RIGHT_FRONT_PAW_INDEX"],
    ["CENTER_SPINE_INDEX", "BASE_NECK_INDEX"],
    ["BASE_NECK_INDEX", "NOSE_INDEX"],
]


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
    set_mixed_map_iters(g_mixed_map_iters)

    kpms_dir = Path(kpms_dir)
    videos_dir = Path(videos_dir)
    poses_csv_dir = Path(poses_csv_dir)
    project_dir = kpms_dir / project_name

    for dir in (kpms_dir, videos_dir, poses_csv_dir):
        if not dir.is_dir():
            raise FileExistsError(f"Directory '{dir}' does not exist.")

    kpms.setup_project(
        str(project_dir),
        video_dir=str(videos_dir),
        bodyparts=G_BODYPARTS,
        skeleton=G_SKELETON,
        overwrite=True,
    )

    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    set_up_logging(log_dir)

    kpms.update_config(
        str(project_dir),
        anterior_bodyparts=["BASE_NECK_INDEX"],
        posterior_bodyparts=["BASE_TAIL_INDEX"],
        use_bodyparts=G_BODYPARTS
    )

    if jax.devices()[0].platform != "cpu":
        print("\n--- GPU USAGE ---\n")
        print_gpu_usage()

    print("\n--- DATA LOADING + FORMATTING ---")
    data, metadata, coords = load_and_format_data(str(poses_csv_dir), str(project_dir))

    print("\n--- PCA ANALYSIS ---")
    config_fn = lambda: kpms.load_config(str(project_dir))
    pca, n_comp = perform_pca(data, config_fn, str(project_dir))
    print(f"Components explaining >90% variance: {n_comp}")
    kpms.update_config(str(project_dir), latent_dim=n_comp)

    print("\n--- FITTING AR-HMM ---")
    fit_and_save_model(
        model_name,
        data,
        metadata,
        pca,
        config_fn,
        project_dir,
        full_model_iters = g_full_model_iters,
        arhmm_iters      = g_arhmm_iters,
        kappa            = g_kappa,
        reduced_kappa    = g_reduced_kappa,
        seed             = seed,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters for Keypoint-MoSeq training")

    ### configuration parameter

    parser.add_argument("--project_name", type=str, required=True,
                        help="Name of the keypoint-MoSeq project")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of keypoint-MoSeq model")
    parser.add_argument("--kpms_dir", type=str, required=True,
                        help="Path to write the keypoint-MoSeq directory project to")
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Path that contains the raw videos")
    parser.add_argument("--poses_csv_dir", type=str, required=True,
                        help="Path that contains the pose data in CSV format")

    ### model training parameters

    parser.add_argument("--g_mixed_map_iters", type=int, default=8,
                        help="Degree of serializing computations; reduce if running out of GPU memory")
    parser.add_argument("--g_arhmm_iters", type=int, default=100,
                        help="Number of AR-HMM fitting iterations")
    parser.add_argument("--g_full_model_iters", type=int, default=400,
                        help="Number of full model fitting iterations")
    parser.add_argument("--g_kappa", type=float, default=1e6,
                        help="Stickiness hyperparameter for AR HMM fitting")
    parser.add_argument("--g_reduced_kappa", type=float, default=1e5,
                        help="Stickiness hyperparameter for full model fitting")
    parser.add_argument("--seed", type=int, default=623,
                        help="Seed for model fitting initialization")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---\n")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_name, args.model_name, args.kpms_dir, args.videos_dir, args.poses_csv_dir,
         args.g_mixed_map_iters, args.g_arhmm_iters, args.g_full_model_iters, args.g_kappa,
         args.g_reduced_kappa, args.seed)
