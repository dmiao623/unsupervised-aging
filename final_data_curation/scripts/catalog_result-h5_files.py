"""Given a KPMS project directory, determines which result-h5 files are present. Specifically,
reads all *.h5 files that contain the string "results" in their path. For now, just prints the structure of the files.
"""

import argparse
import os
from datetime import date
from pathlib import Path

import h5py


EXCLUDED_DIRS = {"logs"}


def find_model_dirs(kpms_dir: Path, project_name: str) -> list[str]:
    """Find all model directories under kpms_dir/project_name, excluding certain dirs."""
    project_path = kpms_dir / project_name
    if not project_path.exists():
        raise FileNotFoundError(f"Project directory not found: {project_path}")
    
    model_dirs = [
        d.name for d in project_path.iterdir()
        if d.is_dir() and d.name not in EXCLUDED_DIRS
    ]
    return model_dirs


def find_result_h5_files(kpms_dir: Path, project_name: str, model_name: str) -> list[Path]:
    """Find all .h5 files containing 'result' in their name."""
    search_dir = kpms_dir / project_name / model_name
    if not search_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {search_dir}")
    
    result_files = []
    for h5_file in search_dir.rglob("*.h5"):
        if "result" in h5_file.name.lower():
            result_files.append(h5_file)
    
    return sorted(result_files)


def truncate_path(path: Path, max_parts: int = 4) -> str:
    """Truncate a path to show only the last N parts, with ellipsis if truncated."""
    parts = path.parts
    if len(parts) <= max_parts:
        return str(path)
    return ".../" + "/".join(parts[-max_parts:])


def get_h5_groups(h5_file: Path) -> dict:
    """
    Extract all groups from an HDF5 file.
    
    Returns a dict with:
    - 'groups': list of group paths
    - 'total': total number of groups
    - 'error': error message if any
    """
    result = {
        'groups': [],
        'total': 0,
    }
    
    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            result['groups'].append(name)
    
    try:
        with h5py.File(h5_file, 'r') as f:
            f.visititems(visitor)
            result['total'] = len(result['groups'])
    except Exception as e:
        result['error'] = str(e)
    
    return result


def clean_group_name(name: str) -> str:
    """Clean a group name by removing .csv suffix."""
    if name.endswith(".csv"):
        return name[:-4]
    return name


def format_h5_groups(groups_info: dict, indent: str = "      ", max_groups: int = 20) -> str:
    """Format the h5 groups for display."""
    if 'error' in groups_info:
        return f"{indent}⚠️  Error reading file: {groups_info['error']}"
    
    total = groups_info['total']
    groups = groups_info['groups']
    
    if total == 0:
        return f"{indent}Groups: (none)"
    
    lines = [f"{indent}Groups ({total} total):"]
    
    display_groups = groups[:max_groups]
    for g in display_groups:
        lines.append(f"{indent}  {g}")
    
    if total > max_groups:
        lines.append(f"{indent}  ... and {total - max_groups} more")
    
    return "\n".join(lines)


def print_results(
    results: dict[str, list[Path]],
    truncate: bool = True,
    max_files: int = 5,
    show_contents: bool = True,
) -> set[str]:
    """Print results in a structured, truncated format.
    
    Returns the set of all unique group names across all files.
    """
    all_unique_groups: set[str] = set()
    
    for model_name, files in results.items():
        print(f"\n📁 Model: {model_name}")
        print(f"   Found {len(files)} result h5 file(s)")
        
        if not files:
            continue
        
        display_files = files[:max_files] if truncate and len(files) > max_files else files
        
        # Collect groups from ALL files (not just displayed ones)
        for f in files:
            groups_info = get_h5_groups(f)
            if 'error' not in groups_info:
                all_unique_groups.update(groups_info['groups'])
        
        for f in display_files:
            display_path = truncate_path(f) if truncate else str(f)
            print(f"\n   • {display_path}")
            
            if show_contents:
                groups_info = get_h5_groups(f)
                print(format_h5_groups(groups_info))
        
        if truncate and len(files) > max_files:
            print(f"\n   ... and {len(files) - max_files} more file(s)")
    
    return all_unique_groups


def main():
    default_kpms_dir = os.path.expandvars("${UNSUPERVISED_AGING}/data/kpms_projects/")
    
    parser = argparse.ArgumentParser(
        description="Catalog result .h5 files in a KPMS project directory."
    )
    parser.add_argument(
        "--kpms-dir",
        type=str,
        default=default_kpms_dir,
        help=f"Path to KPMS projects directory (default: {default_kpms_dir})"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Name of the KPMS project"
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default="",
        help="Comma-separated list of model names. If empty, searches for all model directories."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum number of files to display per model (default: 5)"
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Disable path truncation and file limit"
    )
    parser.add_argument(
        "--no-contents",
        action="store_true",
        help="Only list files, don't show h5 file contents"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path for the unique groups file. If empty, uses default naming convention."
    )
    
    args = parser.parse_args()
    
    kpms_dir = Path(args.kpms_dir)
    project_name = args.project_name
    
    # Determine model names
    if args.model_names:
        model_names = [m.strip() for m in args.model_names.split(",")]
    else:
        print(f"No model names provided. Searching for model directories in {kpms_dir / project_name}...")
        model_names = find_model_dirs(kpms_dir, project_name)
        print(f"Found {len(model_names)} model directory(ies): {model_names}")
    
    # Find result h5 files for each model
    results = {}
    for model_name in model_names:
        try:
            files = find_result_h5_files(kpms_dir, project_name, model_name)
            results[model_name] = files
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            results[model_name] = []
    
    # Print results
    truncate = not args.no_truncate
    show_contents = not args.no_contents
    all_unique_groups = print_results(
        results, truncate=truncate, max_files=args.max_files, show_contents=show_contents
    )
    
    # Clean group names (remove .csv suffix)
    cleaned_groups = {clean_group_name(g) for g in all_unique_groups}
    sorted_groups = sorted(cleaned_groups)
    
    # Summary
    total_files = sum(len(f) for f in results.values())
    print(f"\n{'='*50}")
    print(f"Total: {total_files} result h5 file(s) across {len(results)} model(s)")
    print(f"Unique group names across all files: {len(sorted_groups)}")
    if sorted_groups:
        display_groups = sorted_groups[:20]
        print(f"  {display_groups}")
        if len(sorted_groups) > 20:
            print(f"  ... and {len(sorted_groups) - 20} more")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        today = date.today().isoformat()
        models_str = "__".join(model_names)
        output_dir = Path(os.path.expandvars("${UNSUPERVISED_AGING}/final_data_curation/"))
        output_path = output_dir / f"{today}_deduped-result-h5-groups__{project_name}__{models_str}.txt"
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for group in sorted_groups:
            f.write(f"{group}\n")
    
    print(f"\nSaved {len(sorted_groups)} unique group names to: {output_path}")


if __name__ == "__main__":
    main()
