#!/usr/bin/env python3
"""
Merge names from deduped-ds.csv and missing-paths_corrected.txt into a single name registry.

The missing paths are processed by:
- Trimming prefix: /projects/kumar-lab/sabnig/Projects/UnsupervisedAging/nflow/results/
- Trimming suffix: _trimmed_filtered_pose_est_v6.h5
- Replacing / with __
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "2025-12-03_deduped-ds.csv"
MISSING_PATHS_FILE = BASE_DIR / "2025-12-03_missing-paths_corrected.txt"
OUTPUT_FILE = BASE_DIR / "2025-12-04_final-name-registry.txt"

# Constants for path processing
PREFIX = "/projects/kumar-lab/sabnig/Projects/UnsupervisedAging/nflow/results/"
SUFFIX = "_trimmed_filtered_pose_est_v6.h5"


def process_missing_path(path: str) -> str:
    """Process a path by trimming prefix/suffix and replacing / with __."""
    path = path.strip()
    if not path:
        return ""
    
    # Remove prefix
    if path.startswith(PREFIX):
        path = path[len(PREFIX):]
    
    # Remove suffix
    if path.endswith(SUFFIX):
        path = path[:-len(SUFFIX)]
    
    # Replace / with __
    return path.replace("/", "__")


def main():
    # Read names from CSV
    df = pd.read_csv(CSV_PATH)
    csv_names = set(df["name"].tolist())
    print(f"Names from CSV: {len(csv_names)}")
    
    # Read and process missing paths
    with open(MISSING_PATHS_FILE, "r") as f:
        raw_paths = f.readlines()
    
    processed_names = set()
    for path in raw_paths:
        processed = process_missing_path(path)
        if processed:
            processed_names.add(processed)
    
    print(f"Names from missing paths: {len(processed_names)}")
    
    # Check for overlap
    overlap = csv_names & processed_names
    if overlap:
        print(f"\nERROR: Found {len(overlap)} overlapping names:")
        for name in sorted(overlap):
            print(f"  - {name}")
        raise ValueError("Overlap detected between CSV names and processed missing paths!")
    
    print("No overlap detected - merging sets")
    
    # Combine and sort
    all_names = sorted(csv_names | processed_names)
    print(f"Total unique names: {len(all_names)}")
    
    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for name in all_names:
            f.write(f"{name}\n")
    
    print(f"\nOutput written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()










