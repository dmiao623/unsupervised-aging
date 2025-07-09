"""Script to generate trajectory plots and grid movies from a trained model.

Usage:
    python kpms_generate_trajectories.py \
        --project_name <project_name> \
        --model_name <model_name> \
        --kpms_dir <path_to_kpms_projects> \
        --poses_csv_dir <path_to_pose_csvs> \
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
    kpms_dir = Path(kpms_dir)
    poses_csv_dir = Path(poses_csv_dir)
    project_dir = kpms_dir / project_name

    print("--- LOADING RESULTS ---")
    results = kpms.load_results(project_dir, model_name)

    print("--- LOADING DATA ---")
    config_fn = lambda: kpms.load_config(project_dir)
    _, _, coordinates = load_and_format_data(poses_csv_dir, project_dir)
    
    print("--- GENERATING TRAJECTORY PLOTS ---")
    kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config_fn())

    print("--- GENERATING GRID MOVIES ---")
    kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **config_fn())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectory plots and grid movies")

    parser.add_argument("--project_name", type=str, required=True,
                        help="Name of the keypoint-MoSeq project")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of keypoint-MoSeq model")
    parser.add_argument("--kpms_dir", type=str, required=True,
                        help="Path of the keypoint-MoSeq project directory")
    parser.add_argument("--poses_csv_dir", type=str, required=True,
                        help="Directory containing pose CSV files to process")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_name, args.model_name, args.kpms_dir, args.poses_csv_dir)
