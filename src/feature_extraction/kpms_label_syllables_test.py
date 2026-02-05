"""Generate KPMS trajectory plots and grid movies with keypoint overlays.

Loads trained model results and pose coordinates, then generates
per-syllable trajectory plots and grid movies with keypoint overlays
for visual inspection of syllable assignments.

Usage::

    python kpms_label_syllables_test.py \\
        --project_name <project_name> \\
        --model_name <model_name> \\
        --kpms_dir <path_to_kpms_projects> \\
        --videos_dir <path_to_videos> \\
        --poses_csv_dir <path_to_pose_csvs>
"""

import argparse
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import keypoint_moseq as kpms
from src.methods import load_and_format_data


def main(
    project_name: str,
    model_name: str,
    kpms_dir: str,
    videos_dir: str,
    poses_csv_dir: str,
):
    """Generate trajectory plots and grid movies for a KPMS model.

    Loads model results and formatted pose coordinates, then produces
    per-syllable trajectory plots (without GIFs) and grid movies with
    keypoint overlays.

    Args:
        project_name: Name of the KPMS project (subdirectory of *kpms_dir*).
        model_name: Name of the trained model.
        kpms_dir: Parent directory containing KPMS project directories.
        videos_dir: Directory containing the raw video files.
        poses_csv_dir: Directory containing pose estimation CSVs.
    """
    kpms_dir = Path(kpms_dir)
    videos_dir = Path(videos_dir)
    poses_csv_dir = Path(poses_csv_dir)
    project_dir = kpms_dir / project_name

    print("--- LOADING RESULTS ---")
    results = kpms.load_results(project_dir, model_name)

    print("--- LOADING DATA ---")
    _, _, coordinates = load_and_format_data(poses_csv_dir, project_dir)
    coordinates = {k: v[..., ::-1] for k, v in coordinates.items()}

    config = kpms.load_config(project_dir)
    config["video_dir"] = str(videos_dir)

    print("--- GENERATING TRAJECTORY PLOTS ---")
    kpms.generate_trajectory_plots(
        coordinates,
        results,
        project_dir,
        model_name,
        **config,
        save_gifs=False,
        save_individually=False,
        get_limits_pctl=0.5,
    )

    print("--- GENERATING GRID MOVIES ---")
    kpms.generate_grid_movies(
        results,
        project_dir,
        model_name,
        coordinates=coordinates,
        **config,
        overlay_keypoints=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectory plots and grid movies with keypoint overlays.",
    )

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
        help="Directory containing pose CSV files",
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
    )
