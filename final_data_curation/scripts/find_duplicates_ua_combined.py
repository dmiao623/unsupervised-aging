import argparse
from pathlib import Path

import pandas as pd

def find_duplicates(
    input_csv: Path,
    output_csv: Path,
    key_col: str = "NetworkFilename",
) -> None:
    """
    Identify fully and partially duplicated rows based on a key column.

    - "duplicate_group" is the count of rows sharing the same key_col.
    - "dup_type" is:
        - "full"    -> this row has at least one other row that is identical
                       across all columns (including key_col).
        - "partial" -> this row shares key_col with at least one other row,
                       but there is no completely identical row.
        - "unique"  -> key_col appears only once in the file.
    Only rows with duplicate_group > 1 are considered duplicated; "unique"
    is included mainly for completeness in the output.
    """

    df = pd.read_csv(input_csv)

    if key_col not in df.columns:
        raise ValueError(f"Column '{key_col}' not found in CSV.")

    # Add an index column so you can trace back to the original row number if needed
    df = df.reset_index().rename(columns={"index": "orig_row_index"})

    # How many times each key appears
    key_counts = df[key_col].value_counts()
    df["duplicate_group"] = df[key_col].map(key_counts)

    # Flag rows that are part of any duplicated key
    duplicated_key_mask = df["duplicate_group"] > 1

    # Fully duplicated rows: every column identical to at least one other row
    # (pandas.duplicated with keep=False checks full-row duplicates).
    full_dup_mask = df.duplicated(keep=False)

    # Initialize with "unique"; then overwrite for duplicates
    df["dup_type"] = "unique"
    df.loc[duplicated_key_mask & full_dup_mask, "dup_type"] = "full"
    df.loc[duplicated_key_mask & ~full_dup_mask, "dup_type"] = "partial"

    # Keep only rows that share a key with at least one other row
    duplicated_rows = df[df["duplicate_group"] > 1].copy()

    duplicated_rows.to_csv(output_csv, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify fully and partially duplicated rows in a UA CSV based on NetworkFilename."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV. Defaults to 2025-12-30_UA_combined_masterdf.csv in final_data_curation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV. Defaults to <input_basename>_duplicates.csv in the same directory.",
    )
    parser.add_argument(
        "--key-col",
        type=str,
        default="NetworkFilename",
        help="Column to use as the key for grouping duplicates (default: NetworkFilename).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    base_dir = Path(__file__).resolve().parents[1]

    if args.input is None:
        input_path = base_dir / "2025-12-30_UA_combined_masterdf.csv"
    else:
        input_path = Path(args.input)

    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_duplicates{input_path.suffix}")
    else:
        output_path = Path(args.output)

    find_duplicates(input_path, output_path, key_col=args.key_col)

