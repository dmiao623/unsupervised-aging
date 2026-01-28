"""Create symlinks for pose .h5 files and matching .mp4 videos into separate folders.

This script reads newline-delimited .h5 file paths, then:
1) Links each .h5 into {dest_dir}/poses using name:
   {prefix}{base}.h5 where:
   - For results paths: {prefix} = <group>__<date>__
   - For failed_corners paths: {prefix} = failed_corners__
   - Otherwise: {prefix} = <parent_dir>__
   - {base} is the filename without the suffix "_trimmed_pose_est_v6.h5" if present, else the stem
2) Finds the corresponding .mp4 by replacing "_trimmed_pose_est_v6.h5" with "_trimmed.mp4"
   (if not present, falls back to replacing ".h5" with ".mp4"), and links it into {dest_dir}/videos
   using the same prefix and base, with ".mp4" extension.

Usage:
    python make_symlinks_poses_videos.py \
        --input_list <path_to_text_file_with_h5_paths> \
        --dest_dir <destination_directory> \
        [--relative] \
        [--force] \
        [--dry_run]

Notes:
    Destination subdirectories 'poses' and 'videos' are created if missing.
"""

import argparse
import os
from pathlib import Path

def compute_prefix_and_base(src_h5: Path) -> tuple[str, str]:
    parts = src_h5.parts
    fname = src_h5.name
    if fname.endswith("_trimmed_pose_est_v6.h5"):
        base = fname[:-len("_trimmed_pose_est_v6.h5")]
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
    name = src_h5.name
    if name.endswith("_trimmed_pose_est_v6.h5"):
        vid_name = name.replace("_trimmed_pose_est_v6.h5", "_trimmed.mp4")
    elif name.endswith(".h5"):
        vid_name = name[:-3] + "mp4"
    else:
        vid_name = src_h5.stem + ".mp4"
    return src_h5.with_name(vid_name)

def link_one(src: Path, dst: Path, relative: bool, force: bool, dry_run: bool) -> bool:
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

def main(input_list: str, dest_dir: str, relative: bool, force: bool, dry_run: bool):
    dest = Path(dest_dir)
    poses_dir = dest / "poses"
    videos_dir = dest / "videos"
    poses_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    with open(input_list, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    made = dict(poses=0, videos=0)
    skipped = dict(poses=0, videos=0)
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
    print(f"Done. Created poses: {made['poses']} (skipped {skipped['poses']}), videos: {made['videos']} (skipped {skipped['videos']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symlinks for .h5 poses and matching .mp4 videos into separate subdirectories.")
    parser.add_argument("--input_list", type=str, required=True, help="Path to a text file containing newline-delimited .h5 source file paths")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory containing 'poses' and 'videos'")
    parser.add_argument("--relative", action="store_true", help="Create relative symlinks from destination to sources")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files/symlinks at destination")
    parser.add_argument("--dry_run", action="store_true", help="Print planned links without creating them")
    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:12}: {v}")
    print("------------------\n")

    main(args.input_list, args.dest_dir, args.relative, args.force, args.dry_run)

