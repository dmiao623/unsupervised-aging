"""Script to apply a Keypoint‑MoSeq model for inference.

This script loads a saved KPMS checkpoint, formats new pose CSV data,
and writes an HDF5 file with the inference results.

Usage:
    python apply_kpms_model.py \
        --project_path <path_to_kpms_project> \
        --model_name <checkpoint_name> \
        --poses_csv_dir <path_to_pose_csvs> \
        --results_path <path_to_output_h5>

Note:
    Intended for use after training a KPMS model with train_kpms.py.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import keypoint_moseq as kpms

from src.utils import print_gpu_usage
from src.methods import load_and_format_data


def main(
    project_path: str,
    model_name: str,
    poses_csv_dir: str,
    results_path: str,
):
    project_path = Path(project_path)
    poses_csv_dir = Path(poses_csv_dir)
    results_path = Path(results_path)

    for p in (project_path, poses_csv_dir):
        if not p.exists():
            raise FileNotFoundError(f"Path '{p}' does not exist.")
    if not project_path.is_dir():
        raise NotADirectoryError(f"'{project_path}' is not a directory.")

    print("\n--- CHECKPOINT LOADING ---")
    model = kpms.load_checkpoint(str(project_path), model_name)[0]

    if jax.devices()[0].platform != "cpu":
        print("\n--- GPU USAGE ---\n")
        print_gpu_usage()

    print("\n--- DATA LOADING + FORMATTING ---")
    data, metadata, _ = load_and_format_data(str(poses_csv_dir), str(project_path))

    print("\n--- MODEL INFERENCE ---")
    config_fn = lambda: kpms.load_config(str(project_path))
    kpms.apply_model(
        model,
        data,
        metadata,
        str(project_path),
        model_name,
        **config_fn(),
        results_path=str(results_path))

    print(f"Inference results written to '{results_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a trained KPMS model to new pose data")

    ### configuration parameter

    parser.add_argument("--project_path", type=str, required=True,
                        help="Path to the KPMS project directory")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the KPMS checkpoint to load")
    parser.add_argument("--poses_csv_dir", type=str, required=True,
                        help="Directory containing pose CSV files to process")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path (including filename) for the output HDF5 file")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_path, args.model_name, args.poses_csv_dir, args.results_path)
