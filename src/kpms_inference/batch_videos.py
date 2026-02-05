"""Distribute files into sequentially-numbered sub-folders.

Scans a source directory for regular files and symlinks them into a series
of target folders, each holding at most *files_per_folder* files. Target
folders are named by appending an incrementing integer to a prefix (e.g.
``poses_csv_1``, ``poses_csv_2``, ...).

Usage::

    python batch_videos.py \\
        --source_dir "/my/data/poses_csv" \\
        --target_dir_prefix "/my/data/poses_csv_" \\
        --files_per_folder 7
"""

import argparse
import math
from pathlib import Path


def main(
    source_dir: Path,
    target_dir_prefix: Path,
    files_per_folder: int,
):
    """Symlink files from *source_dir* into numbered sub-folders.

    Each target folder is created by appending its 1-based index to
    *target_dir_prefix*. Existing symlinks at the destination are
    replaced.

    Args:
        source_dir: Directory containing the files to distribute.
        target_dir_prefix: Path prefix for target directories. An integer
            suffix is appended for each batch.
        files_per_folder: Maximum number of files per destination folder.

    Raises:
        FileNotFoundError: If *source_dir* does not exist or is not a
            directory.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"source directory '{source_dir}' does not exist or is not a directory."
        )

    files = sorted(p for p in source_dir.iterdir() if p.is_file())
    total_files = len(files)
    print(f"found {total_files} files in {source_dir}")

    if total_files == 0:
        print("nothing to do: the source directory is empty.")
        return

    total_folders = math.ceil(total_files / files_per_folder)
    print(
        f"creating {total_folders} folders with <= {files_per_folder} files per folder\n"
    )

    for folder_idx in range(1, total_folders + 1):
        target_dir = Path(f"{target_dir_prefix}{folder_idx}")
        target_dir.mkdir(parents=True, exist_ok=True)

        start = (folder_idx - 1) * files_per_folder
        end = min(folder_idx * files_per_folder, total_files)
        chunk = files[start:end]

        print(f"creating folder: {target_dir} with {len(chunk)} files")
        for fp in chunk:
            dest = target_dir / fp.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(fp.resolve())
            print(f"  Linked: {fp.name} → {dest}")

        if folder_idx == total_folders and len(chunk) < files_per_folder:
            print(
                f"last folder contains {len(chunk)} files (less than the full {files_per_folder})"
            )

    print(f"\ndistributed {total_files} files into {total_folders} folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distribute files from one directory into numbered sub-folders"
    )

    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Directory containing the files to distribute",
    )
    parser.add_argument(
        "--target_dir_prefix",
        type=Path,
        required=True,
        help="Prefix for target directories, e.g. '/path/to/poses_csv_'",
    )
    parser.add_argument(
        "--files_per_folder",
        type=int,
        default=7,
        help="Maximum number of files per destination folder (default: 7)",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.source_dir, args.target_dir_prefix, args.files_per_folder)
