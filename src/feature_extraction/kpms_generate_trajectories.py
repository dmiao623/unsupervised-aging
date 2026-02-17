"""Generate trajectory plots and grid movies from a trained KPMS model.

Loads model results and pose coordinates, then produces a syllable
similarity dendrogram, per-syllable trajectory plots, and grid movies
using ``keypoint_moseq`` visualization utilities.

SLURM Template:
    scripts/templates/kpms_generate_trajectories.sh

Usage::

    python kpms_generate_trajectories.py \\
        --project_name <project_name> \\
        --model_name <model_name> \\
        --kpms_dir <path_to_kpms_projects> \\
        --poses_csv_dir <path_to_pose_csvs> \\
        --dendrogram --trajectory_plots --grid_movies
"""

import argparse
import sys
import warnings
from pathlib import Path

import keypoint_moseq as kpms
from src.methods import load_and_format_data
from src.utils import load_keypoints_pd


def main(
    project_name: str,
    model_name: str,
    kpms_dir: str,
    poses_csv_dir: str,
    dendrogram: bool = False,
    trajectory_plots: bool = False,
    grid_movies: bool = False,
    unsafe_load: bool = False,
):
    """Generate visualizations for a trained KPMS model.

    Loads results and formatted pose coordinates, then produces any
    combination of the following (controlled by boolean flags):

    1. A syllable similarity dendrogram.
    2. Per-syllable trajectory plots.
    3. Grid movies of syllable examples.

    Args:
        project_name: Name of the KPMS project (subdirectory of *kpms_dir*).
        model_name: Name of the trained model to visualize.
        kpms_dir: Parent directory containing KPMS project directories.
        poses_csv_dir: Directory containing pose estimation CSVs. A sibling
            ``videos/`` directory is expected for grid movie generation.
        dendrogram: Generate a syllable similarity dendrogram.
        trajectory_plots: Generate per-syllable trajectory plots.
        grid_movies: Generate grid movies of syllable examples.
        unsafe_load: Use ``load_keypoints_pd`` instead of
            ``load_and_format_data`` to load coordinates directly from
            CSVs, bypassing KPMS formatting and validation.
    """
    kpms_dir = Path(kpms_dir)
    poses_csv_dir = Path(poses_csv_dir)
    project_dir = kpms_dir / project_name

    print("--- LOADING RESULTS ---")
    results = kpms.load_results(project_dir, model_name)

    print("--- LOADING DATA ---")
    config_fn = lambda: kpms.load_config(project_dir)
    if unsafe_load:
        coordinates, _ = load_keypoints_pd(str(poses_csv_dir))
    else:
        _, _, coordinates, _ = load_and_format_data(poses_csv_dir, project_dir)

    shared_keys = set(results.keys()) & set(coordinates.keys())
    if not shared_keys:
        sys.exit(
            "ERROR: No shared keys between results and coordinates. "
            "Cannot proceed."
        )

    results_only = set(results.keys()) - shared_keys
    coordinates_only = set(coordinates.keys()) - shared_keys
    if results_only or coordinates_only:
        warnings.warn(
            f"Key mismatch between results and coordinates. "
            f"Results-only: {results_only or '{}'}, "
            f"Coordinates-only: {coordinates_only or '{}'}. "
            f"Continuing with {len(shared_keys)} shared keys."
        )
        results = {k: results[k] for k in shared_keys}
        coordinates = {k: coordinates[k] for k in shared_keys}

    _tmp = config_fn()
    _tmp["video_dir"] = poses_csv_dir / "../videos"

    if dendrogram:
        print("--- GENERATING DENDROGRAM ---")
        kpms.plot_similarity_dendrogram(
            coordinates, results, project_dir, model_name, **_tmp
        )

    if trajectory_plots:
        print("--- GENERATING TRAJECTORY PLOTS ---")
        kpms.generate_trajectory_plots(
            coordinates, results, project_dir, model_name, **_tmp
        )

    if grid_movies:
        print("--- GENERATING GRID MOVIES ---")
        reversed_coordinates = {k: v[..., ::-1] for k, v in coordinates.items()}
        kpms.generate_grid_movies(
            results,
            project_dir,
            model_name,
            coordinates=reversed_coordinates,
            **_tmp,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectory plots and grid movies"
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
        help="Directory containing pose CSV files to process",
    )
    parser.add_argument(
        "--dendrogram",
        action="store_true",
        help="Generate a syllable similarity dendrogram",
    )
    parser.add_argument(
        "--trajectory_plots",
        action="store_true",
        help="Generate per-syllable trajectory plots",
    )
    parser.add_argument(
        "--grid_movies",
        action="store_true",
        help="Generate grid movies of syllable examples",
    )
    parser.add_argument(
        "--unsafe_load",
        action="store_true",
        help="Use load_keypoints_pd to load coordinates directly from CSVs, "
        "bypassing KPMS formatting and validation",
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
        args.poses_csv_dir,
        dendrogram=args.dendrogram,
        trajectory_plots=args.trajectory_plots,
        grid_movies=args.grid_movies,
        unsafe_load=args.unsafe_load,
    )
