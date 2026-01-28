#!/usr/bin/env python3
"""
Cross-reference two CSV files by latent_embedding_mean_0, latent_embedding_mean_1, and latent_embedding_mean_2.
Find rows that don't have one-to-one matches.
"""

import pandas as pd
import numpy as np

# File paths
file1 = "/projects/kumar-lab/miaod/projects/unsupervised-aging/final_data_curation/2026-01-28_UA_DO_masterdf.csv"
file2 = "/projects/kumar-lab/miaod/projects/unsupervised-aging/final_data_curation/2025-12-04_missing_do-embedding.csv"

# Read the CSV files
print("Reading files...")
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print(f"File 1 ({file1}): {len(df1)} rows")
print(f"File 2 ({file2}): {len(df2)} rows")

# Check if the required columns exist
required_cols = ['latent_embedding_mean_0', 'latent_embedding_mean_1', 'latent_embedding_mean_2']
for col in required_cols:
    if col not in df1.columns:
        print(f"ERROR: Column '{col}' not found in file 1")
    if col not in df2.columns:
        print(f"ERROR: Column '{col}' not found in file 2")

# Create a key column for matching
df1['match_key'] = df1[required_cols].apply(lambda x: tuple(x), axis=1)
df2['match_key'] = df2[required_cols].apply(lambda x: tuple(x), axis=1)

# Count matches for each row in df1
print("\nAnalyzing matches...")
df1['match_count'] = df1['match_key'].apply(lambda key: len(df2[df2['match_key'] == key]))
df2['match_count'] = df2['match_key'].apply(lambda key: len(df1[df1['match_key'] == key]))

# Find rows without one-to-one matches
# A one-to-one match means match_count == 1 for both files

# Rows in df1 that don't have exactly one match in df2
df1_no_match = df1[df1['match_count'] == 0]
df1_multiple_match = df1[df1['match_count'] > 1]
df1_one_to_one = df1[df1['match_count'] == 1]

# Rows in df2 that don't have exactly one match in df1
df2_no_match = df2[df2['match_count'] == 0]
df2_multiple_match = df2[df2['match_count'] > 1]
df2_one_to_one = df2[df2['match_count'] == 1]

# For rows with one match, verify they match each other (true one-to-one)
# Check if matched pairs are truly one-to-one
one_to_one_pairs = []
for idx1, row1 in df1_one_to_one.iterrows():
    matching_rows = df2[df2['match_key'] == row1['match_key']]
    if len(matching_rows) == 1:
        idx2 = matching_rows.index[0]
        # Check reverse: does df2 row match only this df1 row?
        reverse_match = df1[df1['match_key'] == matching_rows.iloc[0]['match_key']]
        if len(reverse_match) == 1:
            one_to_one_pairs.append((idx1, idx2))

# Rows in df1 that are NOT in true one-to-one pairs
df1_not_one_to_one = df1[~df1.index.isin([p[0] for p in one_to_one_pairs])]
df2_not_one_to_one = df2[~df2.index.isin([p[1] for p in one_to_one_pairs])]

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nFile 1 ({file1}):")
print(f"  Total rows: {len(df1)}")
print(f"  Rows with 0 matches in file 2: {len(df1_no_match)}")
print(f"  Rows with 1 match in file 2: {len(df1_one_to_one)}")
print(f"  Rows with >1 matches in file 2: {len(df1_multiple_match)}")
print(f"  Rows NOT in one-to-one pairs: {len(df1_not_one_to_one)}")

print(f"\nFile 2 ({file2}):")
print(f"  Total rows: {len(df2)}")
print(f"  Rows with 0 matches in file 1: {len(df2_no_match)}")
print(f"  Rows with 1 match in file 1: {len(df2_one_to_one)}")
print(f"  Rows with >1 matches in file 1: {len(df2_multiple_match)}")
print(f"  Rows NOT in one-to-one pairs: {len(df2_not_one_to_one)}")

print(f"\nTrue one-to-one matches: {len(one_to_one_pairs)}")

print("\n" + "="*80)
print("ROWS WITHOUT ONE-TO-ONE MATCHES")
print("="*80)

print(f"\nFile 1 rows without one-to-one matches ({len(df1_not_one_to_one)} rows):")
if len(df1_not_one_to_one) > 0:
    # Show key columns for identification
    display_cols = ['NetworkFilename'] if 'NetworkFilename' in df1.columns else []
    if 'name' in df1.columns:
        display_cols = ['name']
    display_cols.extend(required_cols)
    display_cols.append('match_count')
    
    # Only show columns that exist
    available_cols = [col for col in display_cols if col in df1_not_one_to_one.columns]
    print(df1_not_one_to_one[available_cols].to_string())
else:
    print("  None")

print(f"\nFile 2 rows without one-to-one matches ({len(df2_not_one_to_one)} rows):")
if len(df2_not_one_to_one) > 0:
    # Show key columns for identification
    display_cols = ['name'] if 'name' in df2.columns else []
    display_cols.extend(required_cols)
    display_cols.append('match_count')
    
    # Only show columns that exist
    available_cols = [col for col in display_cols if col in df2_not_one_to_one.columns]
    print(df2_not_one_to_one[available_cols].to_string())
else:
    print("  None")

# Save detailed results to CSV files
print("\n" + "="*80)
print("Saving detailed results...")
print("="*80)

df1_not_one_to_one.to_csv(
    "/projects/kumar-lab/miaod/projects/unsupervised-aging/final_data_curation/df1_no_one_to_one_match.csv",
    index=False
)
print(f"Saved: df1_no_one_to_one_match.csv ({len(df1_not_one_to_one)} rows)")

df2_not_one_to_one.to_csv(
    "/projects/kumar-lab/miaod/projects/unsupervised-aging/final_data_curation/df2_no_one_to_one_match.csv",
    index=False
)
print(f"Saved: df2_no_one_to_one_match.csv ({len(df2_not_one_to_one)} rows)")

print("\nAnalysis complete!")

