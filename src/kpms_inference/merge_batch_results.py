"""Merges several result HDF5 files (used for parallel inference) into a single results.h5 file

The script assumes each worker wrote an HDF5 whose filename is composed of a common prefix supplied via
`--result_basepath` followed by a 1‑based integer index (e.g. `results_group_1`, `results_group_2`, …). It then
creates `<kpms_dir>/<project_name>/<model_name>/results.h5` and copies every top‑level group from each input file
into the output file.

Usage:
    python merge_batch_results.py 
        --project_name <project_name> \
        --model_name <model_name> \
        --kpms_dir <path_to_kpms_projects> \
        --result_basepath <base_path_to_results> \
        --num_groups <number_of_groups>
"""

import argparse
import h5py
from pathlib import Path
from tqdm import tqdm

def main(
    project_name: str,
    model_name: str,
    kpms_dir: str,
    result_basepath: str,
    num_groups: int,
):
    kpms_dir = Path(kpms_dir)
    project_dir = kpms_dir / project_name

    result_dir = Path(project_dir) / model_name
    result_path = result_dir / "results.h5"
    if result_path.is_file():
        raise ValueError(f"file at {result_path} already exists")

    result_group_files = [Path(f"{result_basepath}{i}") for i in range(1, num_groups+1)]
    with h5py.File(result_path, "w") as h5out:
        for result_group_file in tqdm(result_group_files, desc="combining batched results"):
            with h5py.File(result_group_file, "r") as h5in:
                for item in h5in:
                    if item in h5out:
                        raise ValueError(f"duplicate top-level group '{item}' encountered in {result_group_file.name}")
                    h5in.copy(item, h5out)
    print(f"successfully combined {num_groups} files into {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge result files into a single result.h5")

    parser.add_argument("--project_name", type=str, required=True,
                        help="Name of the keypoint-MoSeq project")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of keypoint-MoSeq model")
    parser.add_argument("--kpms_dir", type=str, required=True,
                        help="Path of the keypoint-MoSeq project directory")
    parser.add_argument("--result_basepath", type=str, required=True,
                        help="Path stem for the HDF5 files to merge")
    parser.add_argument("--num_groups", type=int, required=True,
                        help="The number of result files to merge")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_name, args.model_name, args.kpms_dir, args.result_basepath, args.num_groups)
