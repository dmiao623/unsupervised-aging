"""Checks if missing paths from `2025-11-03_missing-paths.txt` are absent from `2025-12-03_deduped-ds.csv`
and that the corresponding metadata is present in `2025-03-10_b6-do_frailty.csv`.
"""

import pandas as pd
from pathlib import Path

def main():
    # Define file paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent
    
    missing_paths_parsed_file = data_dir / "2025-11-03_missing-paths_parsed.txt"
    missing_paths_full_file = data_dir / "2025-11-03_missing-paths.txt"
    frailty_csv = data_dir / "2025-03-10_b6-do_frailty.csv"
    deduped_csv = data_dir / "2025-12-03_deduped-ds.csv"
    
    # Read parsed paths (used for processing)
    with open(missing_paths_parsed_file, "r") as f:
        missing_paths = [line.strip() for line in f if line.strip()]
    
    # Read full paths (used for printing)
    with open(missing_paths_full_file, "r") as f:
        missing_paths_full = [line.strip() for line in f if line.strip()]
    
    # Create mapping from parsed path to full path
    parsed_to_full = dict(zip(missing_paths, missing_paths_full))
    
    # Read CSVs
    frailty_df = pd.read_csv(frailty_csv)
    deduped_df = pd.read_csv(deduped_csv)
    
    # Get NetworkFilename values from frailty CSV
    network_filenames = set(frailty_df["NetworkFilename"].values)
    
    # Get name values from deduped CSV
    deduped_names = set(deduped_df["name"].values)
    
    # Check 1: Report missing paths NOT present in frailty CSV's NetworkFilename column
    print("=" * 60)
    print("Missing paths NOT found in frailty CSV (NetworkFilename):")
    print("=" * 60)
    not_in_frailty = []
    for path in missing_paths:
        if path not in network_filenames:
            not_in_frailty.append(path)
            print(f"  {parsed_to_full[path]}")
    if not not_in_frailty:
        print("  (All missing paths are present in frailty CSV)")
    print(f"\nTotal: {len(not_in_frailty)} / {len(missing_paths)} not found\n")
    
    # Check 2: Report missing paths that ARE present in deduped CSV's name column
    # Preprocess: remove ".avi" suffix and replace "/" with "__"
    print("=" * 60)
    print("Missing paths FOUND in deduped CSV (name column):")
    print("(These should NOT be present if they are truly missing)")
    print("=" * 60)
    found_in_deduped = []
    for path in missing_paths:
        # Preprocess: remove .avi and replace / with __
        processed_name = path.replace(".avi", "").replace("/", "__")
        if processed_name in deduped_names:
            found_in_deduped.append((path, processed_name))
            print(f"  {parsed_to_full[path]}")
            print(f"    -> Found as: {processed_name}")
    if not found_in_deduped:
        print("  (None of the missing paths are in deduped CSV - this is expected)")
    print(f"\nTotal: {len(found_in_deduped)} / {len(missing_paths)} found (unexpected)\n")
    
    # Create set of paths found in deduped for quick lookup
    found_in_deduped_paths = set(path for path, _ in found_in_deduped)
    
    # Identify correctly formatted paths: present in frailty CSV but missing from deduped CSV
    corrected_paths = []
    for path in missing_paths:
        in_frailty = path in network_filenames
        in_deduped = path in found_in_deduped_paths
        if in_frailty and not in_deduped:
            corrected_paths.append(parsed_to_full[path])
    
    # Write corrected paths to file
    output_file = data_dir / "2025-12-03_missing-paths_corrected.txt"
    with open(output_file, "w") as f:
        for full_path in corrected_paths:
            f.write(full_path + "\n")
    
    print("=" * 60)
    print(f"Corrected missing paths written to: {output_file.name}")
    print(f"Total: {len(corrected_paths)} paths (in metadata, not in deduped)")
    print("=" * 60)


if __name__ == "__main__":
    main() 
