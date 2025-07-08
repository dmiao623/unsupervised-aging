"""Converts pose-estimation H5 files to CSV inside a dataset directory.

This script scans the sub-directory ``<dataset_dir>/poses`` for H5 files (e.g., `*_pose_est_v6.h5`).
Each file is converted to a CSV with the same base name and written to `<dataset_dir>/poses_csv` by
calling `src.preprocessing.h5_to_csv_poses`.

Usage:
    python convert_pose_h5_to_csv.py \
        --dataset_dir <path_to_dataset_dir> \
        [--strict_mode]
"""

import argparse

from pathlib import Path

from src.preprocessing import h5_to_csv_poses

def main(
    dataset_dir: Path,
    strict_mode: bool
):
    pose_dir = dataset_dir / "poses"
    pose_csv_dir = dataset_dir / "poses_csv"

    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory {pose_dir} does not exist.")

    pose_csv_dir.mkdir(exist_ok=True)
    if strict_mode and any(pose_csv_dir.iterdir()):
        raise ValueError(f"Pose CSV directory {pose_csv_dir} is not empty.")
    h5_to_csv_poses(pose_dir, pose_csv_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pose H5 files to CSV within a dataset directory")

    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the dataset directory containing a 'poses' subdirectory with .h5 files")
    parser.add_argument("--strict_mode", action="store_true",
                        help="If set, abort if the output 'poses_csv' directory is not empty")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(Path(args.dataset_dir), args.strict_mode)
