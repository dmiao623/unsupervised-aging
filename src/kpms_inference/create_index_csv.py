"""Create an ``index.csv`` for Keypoint-MoSeq downstream analysis.

Keypoint-MoSeq expects a project-level ``index.csv`` containing group
information. When groups are not well-defined, this script creates
placeholder groups where each animal is assigned its own group. See the
`Keypoint-MoSeq docs <https://keypoint-moseq.readthedocs.io/en/latest/
analysis.html#assign-groups>`_ for use cases.

Usage::

    python create_index_csv.py \\
        --project_name <project_name> \\
        --kpms_dir <path_to_kpms_projects> \\
        --poses_csv_dir <path_to_pose_csvs>
"""

import argparse
from pathlib import Path


def main(
    project_name: str,
    kpms_dir: str,
    poses_csv_dir: str,
):
    """Write an ``index.csv`` mapping each pose file to its own group.

    Iterates over all entries in *poses_csv_dir* and writes a two-column
    CSV (``name``, ``group``) into the project directory.

    Args:
        project_name: Name of the KPMS project (subdirectory of *kpms_dir*).
        kpms_dir: Parent directory containing KPMS project directories.
        poses_csv_dir: Directory containing pose CSV files.

    Raises:
        FileExistsError: If ``index.csv`` already exists in the project
            directory.
    """
    kpms_dir = Path(kpms_dir)
    poses_csv_dir = Path(poses_csv_dir)
    project_dir = kpms_dir / project_name

    index_path = project_dir / "index.csv"
    if index_path.is_file():
        raise FileExistsError(f"file at {index_path} already exists")

    csv_header = "name,group"
    csv_rows = "\n".join(
        [f"{file.name},{file.stem}" for file in poses_csv_dir.iterdir()]
    )
    csv_text = csv_header + "\n" + csv_rows

    index_path.write_text(csv_text)
    print(f"wrote group data to {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create index.csv for Keypoint-MoSeq")

    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the keypoint-MoSeq project",
    )
    parser.add_argument(
        "--kpms_dir",
        type=str,
        required=True,
        help="Path of the keypoint-MoSeq project directory",
    )
    parser.add_argument(
        "--poses_csv_dir",
        type=str,
        required=True,
        help="Directory containing pose CSV files to process",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_name, args.kpms_dir, args.poses_csv_dir)
