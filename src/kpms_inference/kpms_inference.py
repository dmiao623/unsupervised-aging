"""Script to apply a Keypoint‑MoSeq model for inference.

This script loads a saved KPMS checkpoint, formats new pose CSV data,
and writes an HDF5 file with the inference results.

Usage:
    python kpms_inference.py \
        --project_name <project_name> \
        --model_name <model_name> \
        --kpms_dir <path_to_kpms_projects> \
        --poses_csv_dir <path_to_pose_csvs> \
        --result_path <path_to_output_h5>

Note:
    Intended for use after training a KPMS model with train_kpms.py.
"""

import argparse
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import keypoint_moseq as kpms

from src.utils import print_gpu_usage
from src.methods import load_and_format_data


def main(
    project_name: str,
    model_name: str,
    kpms_dir: str,
    poses_csv_dir: str,
    result_path: str,
):
    kpms_dir = Path(kpms_dir)
    project_dir = kpms_dir / project_name
    poses_csv_dir = Path(poses_csv_dir)
    result_path = Path(result_path)

    for p in (project_dir, poses_csv_dir):
        if not p.exists():
            raise FileNotFoundError(f"Path '{p}' does not exist.")
    if not project_dir.is_dir():
        raise NotADirectoryError(f"'{project_dir}' is not a directory.")

    print("\n--- CHECKPOINT LOADING ---")
    model = kpms.load_checkpoint(str(project_dir), model_name)[0]

    if jax.devices()[0].platform != "cpu":
        print("\n--- GPU USAGE ---\n")
        print_gpu_usage()

    print("\n--- DATA LOADING + FORMATTING ---")
    data, metadata, _ = load_and_format_data(str(poses_csv_dir), str(project_dir))

    print("\n--- MODEL INFERENCE ---")
    config_fn = lambda: kpms.load_config(str(project_dir))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    kpms.apply_model(model, data, metadata, str(project_dir), model_name,**config_fn(),
                     result_path=str(result_path))

    print(f"inference results written to '{result_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a trained KPMS model to new pose data")

    parser.add_argument("--project_name", type=str, required=True,
                        help="Name of the keypoint-MoSeq project")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of keypoint-MoSeq model")
    parser.add_argument("--kpms_dir", type=str, required=True,
                        help="Path of the keypoint-MoSeq project directory")
    parser.add_argument("--poses_csv_dir", type=str, required=True,
                        help="Directory containing pose CSV files to process")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path (including filename) for the output HDF5 file")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_name, args.model_name, args.kpms_dir, args.poses_csv_dir, args.result_path)
