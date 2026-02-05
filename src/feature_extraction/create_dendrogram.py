"""Compute and plot a syllable similarity dendrogram from KPMS results.

Loads trained model results and pose coordinates, then uses
``kpms.plot_similarity_dendrogram`` to compute pairwise syllable
distances and save the resulting dendrogram figure.

Usage::

    python create_dendrogram.py \\
        --project_name <project_name> \\
        --model_name <model_name> \\
        --kpms_dir <path_to_kpms_projects> \\
        --poses_csv_dir <path_to_pose_csvs>
"""

import argparse
from pathlib import Path

import keypoint_moseq as kpms
from src.methods import load_and_format_data


def main(
    project_name: str,
    model_name: str,
    kpms_dir: str,
    poses_csv_dir: str,
):
    """Compute syllable similarity and save a dendrogram plot.

    Loads model results and formatted pose coordinates, computes
    pairwise syllable distances, and writes the dendrogram figure into
    the model directory via ``kpms.plot_similarity_dendrogram``.

    Args:
        project_name: Name of the KPMS project (subdirectory of *kpms_dir*).
        model_name: Name of the trained model.
        kpms_dir: Parent directory containing KPMS project directories.
        poses_csv_dir: Directory containing pose estimation CSVs.
    """
    kpms_dir = Path(kpms_dir)
    poses_csv_dir = Path(poses_csv_dir)
    project_dir = kpms_dir / project_name

    print("--- LOADING RESULTS ---")
    results = kpms.load_results(project_dir, model_name)

    print("--- LOADING DATA ---")
    _, _, coordinates = load_and_format_data(poses_csv_dir, project_dir)
    coordinates = {k: v[..., ::-1] for k, v in coordinates.items()}

    print("--- GENERATING DENDROGRAM ---")
    config = kpms.load_config(project_dir)
    kpms.plot_similarity_dendrogram(
        coordinates,
        results,
        project_dir,
        model_name,
        **config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot a syllable similarity dendrogram.",
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

    main(args.project_name, args.model_name, args.kpms_dir, args.poses_csv_dir)
