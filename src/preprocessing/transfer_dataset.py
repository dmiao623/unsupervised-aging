"""Transfer trimmed video and pose-estimation files into a dataset directory.

Reads a ``metadata.csv`` located in *dataset_dir* to obtain identifiers for
each recording. Identifiers are stored in a column named ``name`` and use
double underscores (``"__"``) instead of forward slashes to encode
sub-directories. The script reconstructs full paths under *original_dir*,
verifies that every source file exists, and creates symbolic links inside
``<dataset_dir>/videos`` and ``<dataset_dir>/poses``.

Usage::

    python transfer_dataset.py \\
        --original_dir <path_to_original_dir> \\
        --dataset_dir <path_to_dataset_dir>
"""

import argparse
import pandas as pd

from pathlib import Path


def main(
    original_dir: Path,
    dataset_dir: Path,
):
    """Symlink trimmed videos and pose H5 files into the dataset directory.

    For each row in ``<dataset_dir>/metadata.csv``, the script derives the
    source paths by replacing ``"__"`` with ``"/"`` and appending the expected
    suffixes (``_trimmed.mp4`` for videos, ``_trimmed_pose_est_v6.h5`` for
    poses). After verifying all sources exist, it symlinks them into
    ``<dataset_dir>/videos`` and ``<dataset_dir>/poses``.

    Args:
        original_dir: Root directory containing the original video and pose
            files, organized in subdirectories.
        dataset_dir: Dataset directory containing ``metadata.csv`` and where
            ``videos/`` and ``poses/`` subdirectories will be created.

    Raises:
        ValueError: If any expected source file does not exist.
    """
    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    names = metadata_df["name"].tolist()

    video_src = [original_dir / f"{n.replace('__', '/')}_trimmed.mp4" for n in names]
    pose_src = [
        original_dir / f"{n.replace('__', '/')}_trimmed_pose_est_v6.h5" for n in names
    ]

    for f in video_src + pose_src:
        if not f.is_file():
            raise ValueError(f"{f} does not exist.")

    print(f"Beginning transfer of {len(names)} video and pose files.")

    for src_list, category in [(video_src, "videos"), (pose_src, "poses")]:
        (dataset_dir / category).mkdir(exist_ok=True)
        for src, n in zip(src_list, names):
            dst = dataset_dir / category / f"{n}{src.suffix}"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)

    print("File transfer successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer trimmed video and pose files into a dataset directory.",
    )

    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Path to the directory containing the original video and pose files",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory where symlinks will be created",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(Path(args.original_dir), Path(args.dataset_dir))
