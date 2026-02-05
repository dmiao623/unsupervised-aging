"""Merge batched HDF5 inference results into a single ``results.h5``.

Assumes each parallel worker wrote an HDF5 file named ``result-<i>.h5``
inside a common directory. All top-level groups are copied into one
output file; duplicate keys cause an error.

Usage::

    python merge_batch_results.py \\
        --project_name <project_name> \\
        --model_name <model_name> \\
        --kpms_dir <path_to_kpms_projects> \\
        --group_result_dir <base_path_to_group_results> \\
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
    """Combine per-worker HDF5 result files into one ``results.h5``.

    Reads ``result-1.h5`` through ``result-<num_groups>.h5`` from
    *group_result_dir* and copies every top-level group into a single
    output file under ``<project_dir>/<model_name>/results.h5``.

    Args:
        project_name: Name of the KPMS project (subdirectory of *kpms_dir*).
        model_name: Name of the model whose results are being merged.
        kpms_dir: Parent directory containing KPMS project directories.
        group_result_dir: Directory containing the per-worker HDF5 files.
        num_groups: Number of result files to merge.

    Raises:
        FileExistsError: If ``results.h5`` already exists at the output
            path.
        ValueError: If a duplicate top-level HDF5 group is encountered.
    """
    kpms_dir = Path(kpms_dir)
    project_dir = kpms_dir / project_name
    group_result_dir = Path(group_result_dir)

    result_dir = Path(project_dir) / model_name
    result_path = result_dir / "results.h5"
    if result_path.is_file():
        raise FileExistsError(f"file at {result_path} already exists")

    result_group_files = [
        group_result_dir / f"result-{i}.h5" for i in range(1, num_groups + 1)
    ]
    with h5py.File(result_path, "w") as h5out:
        for result_group_file in tqdm(
            result_group_files, desc="combining batched results"
        ):
            with h5py.File(result_group_file, "r") as h5in:
                for item in h5in:
                    if item in h5out:
                        raise ValueError(
                            f"duplicate top-level group '{item}' encountered in {result_group_file.name}"
                        )
                    h5in.copy(item, h5out)
    print(f"successfully combined {num_groups} files into {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge result files into a single result.h5"
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
        "--group_result_dir",
        type=str,
        required=True,
        help="Directory to the HDF5 files to merge",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        required=True,
        help="The number of result files to merge",
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
        args.group_result_dir,
        args.num_groups,
    )
