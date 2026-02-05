"""Convert pose-estimation H5 files to CSV inside a dataset directory.

Scans ``<dataset_dir>/poses`` for H5 files and converts each to a CSV with
the same base name, writing results to ``<dataset_dir>/poses_csv``. The
conversion is handled by ``src.preprocessing.h5_to_csv_poses``.

SLURM Template:
    scripts/templates/convert_h5_to_csv.sh

Usage::

    python convert_h5_to_csv.py \\
        --dataset_dir <path_to_dataset_dir> \\
        [--strict_mode]
"""

import argparse
from pathlib import Path

from src.preprocessing import h5_to_csv_poses


def main(
    dataset_dir: Path,
    strict_mode: bool,
):
    """Convert all H5 pose files in a dataset directory to CSV.

    Reads H5 files from ``<dataset_dir>/poses``, creates the output
    directory ``<dataset_dir>/poses_csv`` if needed, and writes one CSV
    per H5 file.

    Args:
        dataset_dir: Root dataset directory containing a ``poses``
            subdirectory with ``.h5`` files.
        strict_mode: If ``True``, abort when the output ``poses_csv``
            directory already contains files.

    Raises:
        FileNotFoundError: If the ``poses`` subdirectory does not exist.
        ValueError: If *strict_mode* is set and ``poses_csv`` is not empty.
    """
    pose_dir = dataset_dir / "poses"
    pose_csv_dir = dataset_dir / "poses_csv"

    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory {pose_dir} does not exist.")

    pose_csv_dir.mkdir(exist_ok=True)
    if strict_mode and any(pose_csv_dir.iterdir()):
        raise ValueError(f"Pose CSV directory {pose_csv_dir} is not empty.")
    h5_to_csv_poses(pose_dir, pose_csv_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert pose H5 files to CSV within a dataset directory.",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing a 'poses' subdirectory with .h5 files",
    )
    parser.add_argument(
        "--strict_mode",
        action="store_true",
        help="If set, abort if the output 'poses_csv' directory is not empty",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(Path(args.dataset_dir), args.strict_mode)
