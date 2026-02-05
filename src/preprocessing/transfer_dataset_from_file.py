"""Create symlinks for pose H5 files and matching videos into a dataset directory.

Reads newline-delimited ``.h5`` file paths from a text file and creates
symbolic links in ``<dest_dir>/poses`` and ``<dest_dir>/videos``. For each
H5 path the script:

1. Derives a prefix/base pair from the directory structure (see
   ``compute_prefix_and_base``).
2. Links the H5 file as ``<prefix><base>.h5`` into ``<dest_dir>/poses``.
3. Derives the corresponding ``.mp4`` path and links it as
   ``<prefix><base>.mp4`` into ``<dest_dir>/videos``.

Usage::

    python transfer_dataset_from_file.py \\
        --input_list <path_to_text_file_with_h5_paths> \\
        --dest_dir <destination_directory> \\
        [--relative] \\
        [--force] \\
        [--dry_run]

Note:
    Destination subdirectories ``poses`` and ``videos`` are created if
    missing.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple


def compute_prefix_and_base(src_h5: Path) -> Tuple[str, str]:
    """Derive a unique prefix and base name from an H5 source path.

    The prefix encodes the parent directory structure so that files from
    different experiments do not collide when placed in a flat directory.

    * If the path contains a ``results`` component, the prefix is
      ``<group>__<date>__``.
    * If it contains ``failed_corners``, the prefix is
      ``failed_corners__``.
    * Otherwise the immediate parent directory name is used.

    The base name is the filename stripped of the
    ``_trimmed_pose_est_v6.h5`` suffix when present, or the ``.h5``
    extension otherwise.

    Args:
        src_h5: Path to the source H5 file.

    Returns:
        A ``(prefix, base)`` tuple of strings.
    """
    parts = src_h5.parts
    fname = src_h5.name
    if fname.endswith("_trimmed_pose_est_v6.h5"):
        base = fname[: -len("_trimmed_pose_est_v6.h5")]
    elif fname.endswith(".h5"):
        base = fname[:-3]
    else:
        base = src_h5.stem

    if "results" in parts:
        i = parts.index("results")
        if i + 3 <= len(parts):
            group_ = parts[i + 1]
            date_ = parts[i + 2]
            prefix = f"{group_}__{date_}__"
        else:
            prefix = f"{src_h5.parent.name}__"
    elif "failed_corners" in parts:
        prefix = "failed_corners__"
    else:
        prefix = f"{src_h5.parent.name}__"

    return prefix, base


def derive_video_path(src_h5: Path) -> Path:
    """Return the expected video path corresponding to a pose H5 file.

    Replaces the ``_trimmed_pose_est_v6.h5`` suffix with
    ``_trimmed.mp4``. Falls back to swapping ``.h5`` for ``.mp4``.

    Args:
        src_h5: Path to the source H5 file.

    Returns:
        Path to the corresponding video file.
    """
    name = src_h5.name
    if name.endswith("_trimmed_pose_est_v6.h5"):
        vid_name = name.replace("_trimmed_pose_est_v6.h5", "_trimmed.mp4")
    elif name.endswith(".h5"):
        vid_name = name[:-3] + ".mp4"
    else:
        vid_name = src_h5.stem + ".mp4"
    return src_h5.with_name(vid_name)


def link_one(
    src: Path,
    dst: Path,
    relative: bool,
    force: bool,
    dry_run: bool,
) -> bool:
    """Create a single symlink from *dst* to *src*.

    Args:
        src: Source file path.
        dst: Destination symlink path.
        relative: If ``True``, create a relative symlink.
        force: If ``True``, overwrite an existing file or symlink at *dst*.
        dry_run: If ``True``, print the planned action without creating
            the link.

    Returns:
        ``True`` if a link was (or would be) created, ``False`` if
        skipped.
    """
    if relative:
        src_for_link = os.path.relpath(str(src), start=str(dst.parent))
    else:
        src_for_link = str(src)

    if dst.exists() or dst.is_symlink():
        if force:
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
        else:
            print(f"SKIP {dst} (exists)")
            return False

    if dry_run:
        print(f"LINK {src_for_link} -> {dst}")
        return True

    os.symlink(src_for_link, dst)
    print(f"LINKED {src_for_link} -> {dst}")
    return True


def main(
    input_list: str,
    dest_dir: str,
    relative: bool,
    force: bool,
    dry_run: bool,
):
    """Read H5 paths from a file and symlink poses and videos into *dest_dir*.

    For each line in *input_list* the script creates two symlinks: one for
    the pose H5 file and one for the matching video. Counts of created and
    skipped links are printed when finished.

    Args:
        input_list: Path to a text file containing newline-delimited H5
            source file paths. Lines starting with ``#`` are ignored.
        dest_dir: Destination directory. ``poses/`` and ``videos/``
            subdirectories are created automatically.
        relative: Create relative symlinks instead of absolute ones.
        force: Overwrite existing files or symlinks at the destination.
        dry_run: Print planned links without actually creating them.
    """
    dest = Path(dest_dir)
    poses_dir = dest / "poses"
    videos_dir = dest / "videos"
    poses_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    with open(input_list, "r") as f:
        lines = [
            ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")
        ]

    made = {"poses": 0, "videos": 0}
    skipped = {"poses": 0, "videos": 0}

    for line in lines:
        src_h5 = Path(line)
        prefix, base = compute_prefix_and_base(src_h5)

        dst_h5 = poses_dir / f"{prefix}{base}.h5"
        ok = link_one(src_h5, dst_h5, relative, force, dry_run)
        made["poses"] += int(ok)
        skipped["poses"] += int(not ok)

        src_mp4 = derive_video_path(src_h5)
        dst_mp4 = videos_dir / f"{prefix}{base}.mp4"
        ok = link_one(src_mp4, dst_mp4, relative, force, dry_run)
        made["videos"] += int(ok)
        skipped["videos"] += int(not ok)

    print(
        f"Done. Created poses: {made['poses']} (skipped {skipped['poses']}), "
        f"videos: {made['videos']} (skipped {skipped['videos']})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create symlinks for .h5 poses and matching .mp4 videos.",
    )

    parser.add_argument(
        "--input_list",
        type=str,
        required=True,
        help="Path to a text file containing newline-delimited .h5 source file paths",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        required=True,
        help="Destination directory containing 'poses' and 'videos'",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Create relative symlinks from destination to sources",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files/symlinks at destination",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned links without creating them",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.input_list, args.dest_dir, args.relative, args.force, args.dry_run)
