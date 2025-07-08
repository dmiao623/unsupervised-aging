"""Transfers trimmed video and pose-estimation files into a dataset directory via symbolic links.

This script reads a `metadata.csv` located in `dataset_dir` to obtain identifiers for each
recording. Identifiers are stored in a column named `name` and use double underscores ("__")
instead of forward slashes to encode sub-directories. The script reconstructs the full paths
under `original_dir`, appends the expected file suffixes for the trimmed video and the
corresponding pose-estimation results, verifies that every source file exists, and then
creates symbolic links inside `dataset_dir/videos` and `dataset_dir/poses`.

Usage:
    python transfer_trimmed_files.py \
        --original_dir <path_to_original_dir> \
        --dataset_dir <path_to_dataset_dir>
"""

import pandas as pd

from pathlib import Path

def main(
    original_dir: Path,
    dataset_dir: Path,
):
    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    file_basenames = [str(original_dir / name.replace("__", "/")) for name in metadat_df["name"]]

    def str_append(s: str): return lambda x: x + s
    video_files = list(map(str_append("__trimmed.mp4"), file_basenames))
    pose_files  = list(map(str_append("_trimmed_pose_est_v6.h5"), file_basenames))

    for file in (video_files + pose_files):
        if not Path(file).is_file():
            raise ValueError(f"Path {file} does not exist.")
    print(f"Beginning transfer of {len(file_basenames)} videon and pose files.")

    for file_list, category, extension in [(video_files, "videos"), (pose_files, "poses")]:
        basepath = dataset_dir / category
        basepath.mkdir(exists_ok=True)

        for original_file, basename in zip(file_list, file_basenames):
            original_path = Path(original_file)
            new_path = basepath / (basenames + original_path.suffix)
            new_path.symlink_to(original_path)
    print(f"File transfer successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer trimmed video and pose files into dataset directory")

    parser.add_argument("--original_dir", type=str, required=True,
                        help="Path to the directory containing the original video and pose files")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the dataset directory where symlinks will be created")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(Path(args.original_dir), Path(args.dataset_dir))
