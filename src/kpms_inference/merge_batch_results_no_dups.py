"""
Merges several result HDF5 files (used for parallel inference) into a single results.h5 file

The script assumes each worker wrote an HDF5 whose filename is of the form "result-i.h5" and are
inside a common directory.

Usage:
    python merge_batch_results.py 
        --project_name <project_name> \
        --model_name <model_name> \
        --kpms_dir <path_to_kpms_projects> \
        --group_result_dir <base_path_to_group_results> \
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
    group_result_dir: str,
    num_groups: int,
):
    kpms_dir = Path(kpms_dir)
    project_dir = kpms_dir / project_name
    group_result_dir = Path(group_result_dir)

    result_dir = Path(project_dir) / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "results.h5"
    if result_path.is_file():
        raise ValueError(f"file at {result_path} already exists")

    result_group_files = [group_result_dir / f"result-{i}.h5" for i in range(1, num_groups+1)]
    with h5py.File(result_path, "w") as h5out:
        for result_group_file in tqdm(result_group_files, desc="combining batched results"):
            with h5py.File(result_group_file, "r") as h5in:
                for key in h5in:
                    if key in h5out:
                        print(f"warning: duplicate top-level key '{key}' in {result_group_file.name}; keeping existing")
                        continue
                    h5in.copy(key, h5out)
    print(f"successfully combined {num_groups} files into {result_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project_name", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--kpms_dir", required=True)
    p.add_argument("--group_result_dir", required=True)
    p.add_argument("--num_groups", type=int, required=True)
    args = p.parse_args()
    main(args.project_name, args.model_name, args.kpms_dir, args.group_result_dir, args.num_groups)

