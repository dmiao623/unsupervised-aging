"""Concatenates multiple CSV files into combined datasets.

This script takes a list of CSV file paths, loads them into pandas DataFrames, 
and concatenates them into a single DataFrame. It prints informative status 
messages about each file's number of entries and the total combined size. 
The resulting concatenated DataFrame is saved as a new CSV file.

Usage:
    python concat_dataframes.py \
        --input_csvs <list_of_csv_paths> \
        --output_csv <path_to_output_csv>

Notes:
    All input CSVs should have compatible column structures for concatenation. 
    The script outputs a summary of each CSV’s size and the total combined entries.
"""

import argparse
import pandas as pd
import os

def load_and_concatenate(csv_paths):
    dataframes = []
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

def save_dataframe(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Concatenated CSV saved to '{output_path}' with {len(df)} rows.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files into one.")
    parser.add_argument("--input_csvs", nargs="+", required=True,
                        help="List of input CSV file paths to concatenate")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save concatenated output CSV")
    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    combined_df = load_and_concatenate(args.input_csvs)
    save_dataframe(combined_df, args.output_csv)