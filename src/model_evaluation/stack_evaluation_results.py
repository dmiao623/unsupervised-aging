"""Concatenate multiple evaluation-result CSV files into one.

Loads a list of CSV files, concatenates them row-wise into a single
``DataFrame``, and saves the result to a new CSV. A summary of each
file's row count and the combined total is printed during execution.

Usage::

    python stack_evaluation_results.py \\
        --input_csvs <csv_1> <csv_2> ... \\
        --output_csv <path_to_output_csv>

Note:
    All input CSVs should have compatible column structures for
    concatenation.
"""

import argparse
import os
import pandas as pd

from typing import List, Sequence


def load_and_concatenate(csv_paths: Sequence[str]) -> pd.DataFrame:
    """Load CSV files and concatenate them into a single DataFrame.

    Args:
        csv_paths: Ordered list of CSV file paths to load.

    Returns:
        A concatenated ``DataFrame`` with a reset integer index.
    """
    dataframes: List[pd.DataFrame] = []
    total_rows = 0
    print("\n--- LOADING CSV FILES ---")
    for path in csv_paths:
        df = pd.read_csv(path)
        rows = len(df)
        total_rows += rows
        dataframes.append(df)
        print(f"Loaded '{os.path.basename(path)}' with {rows} rows.")
    print(f"Total combined entries: {total_rows}")
    print("--------------------------\n")
    return pd.concat(dataframes, ignore_index=True)


def save_dataframe(df: pd.DataFrame, output_path: str):
    """Write a DataFrame to CSV without the index.

    Args:
        df: DataFrame to save.
        output_path: Destination file path.
    """
    df.to_csv(output_path, index=False)
    print(f"Concatenated CSV saved to '{output_path}' with {len(df)} rows.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate multiple evaluation-result CSV files into one.",
    )

    parser.add_argument(
        "--input_csvs",
        nargs="+",
        required=True,
        help="List of input CSV file paths to concatenate",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save concatenated output CSV",
    )

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    combined_df = load_and_concatenate(args.input_csvs)
    save_dataframe(combined_df, args.output_csv)
