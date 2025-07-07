"""Script to set up a Keypoint-MoSeq project for training.

This script initializes a new Keypoint-MoSeq project directory, sets configuration options,
and performs PCA to determine the appropriate latent dimensionality.

Usage:
    python train_kpms_setup.py \
        --project_name <project_name> \
        --kpms_dir <path_to_kpms_projects> \
        --videos_dir <path_to_videos> \
        --poses_csv_dir <path_to_pose_csvs>

Note:
    Intended to be used in tandem to train_kpms.py to train (multiple) KPMS models. 
"""

import argparse
import os

from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import keypoint_moseq as kpms

from src.utils import set_up_logging, print_gpu_usage
from src.methods import load_and_format_data, perform_pca


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
    kpms_dir: str,
    videos_dir: str,
    poses_csv_dir: str,
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters for Keypoint-MoSeq training")

    ### configuration parameter

    parser.add_argument("--project_name", type=str, required=True,
                        help="Name of the keypoint-MoSeq project")
    parser.add_argument("--kpms_dir", type=str, required=True,
                        help="Path to write the keypoint-MoSeq directory project to")
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="Path that contains the raw videos")
    parser.add_argument("--poses_csv_dir", type=str, required=True,
                        help="Path that contains the pose data in CSV format")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_name, args.kpms_dir, args.videos_dir, args.poses_csv_dir)
