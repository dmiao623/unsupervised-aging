#!/usr/bin/env python3
"""
Match `2026-01-28_UA_DO_masterdf.csv` (column: NetworkFilename) against
`2025-12-04_missing_do-embedding.csv` (column: name) using a suffix rule:

- processed_name = name with "__" replaced by "/"
- A match exists between a NetworkFilename and a processed_name if they share
  an ending substring (suffix), operationalized as:
    norm_processed_name.endswith(norm_networkfilename) OR
    norm_networkfilename.endswith(norm_processed_name)
  where norm_* strips leading "/" for stability.

Reports:
- number of exact (one-to-one) matches
- one-to-many matches (one NetworkFilename matches multiple names)
- many-to-one matches (multiple NetworkFilename match one name)

Also writes CSVs with detailed match counts and matched pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


MASTER = "/projects/kumar-lab/miaod/projects/unsupervised-aging/final_data_curation/2026-01-28_UA_DO_masterdf.csv"
MISSING = "/projects/kumar-lab/miaod/projects/unsupervised-aging/final_data_curation/2025-12-04_missing_do-embedding.csv"
OUT_DIR = "/projects/kumar-lab/miaod/projects/unsupervised-aging/final_data_curation"


def norm(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lstrip("/")


def process_name(name: str) -> str:
    # Per user instruction: replace "__" with "/"
    return norm(str(name).replace("__", "/"))


def suffix_match(a: str, b: str) -> bool:
    # match if some suffix is shared, implemented as "one is suffix of the other"
    if not a or not b:
        return False
    return a.endswith(b) or b.endswith(a)


@dataclass(frozen=True)
class Pair:
    master_idx: int
    missing_idx: int


def main() -> None:
    df_master = pd.read_csv(MASTER, usecols=["NetworkFilename"])
    df_missing = pd.read_csv(MISSING, usecols=["name"])

    df_master["network_norm"] = df_master["NetworkFilename"].map(norm)
    df_missing["name_processed"] = df_missing["name"].map(process_name)

    # Index missing rows by their final path component to narrow search space.
    # Since NetworkFilename is typically "DO....", the last component match is the main signal.
    df_missing["name_tail"] = df_missing["name_processed"].str.split("/").str[-1]
    tail_to_missing_idxs: Dict[str, List[int]] = {}
    for idx, tail in df_missing["name_tail"].items():
        tail_to_missing_idxs.setdefault(str(tail), []).append(idx)

    master_to_missing: Dict[int, Set[int]] = {i: set() for i in df_master.index}
    missing_to_master: Dict[int, Set[int]] = {i: set() for i in df_missing.index}

    pairs: List[Pair] = []

    for mi, network in df_master["network_norm"].items():
        # Candidate missing rows: those whose tail equals this network string, plus a fallback
        # to suffix scan if needed (rare).
        candidates = tail_to_missing_idxs.get(str(network), [])

        matched_any = False
        for ni in candidates:
            pname = df_missing.at[ni, "name_processed"]
            if suffix_match(network, pname):
                matched_any = True
                master_to_missing[mi].add(ni)
                missing_to_master[ni].add(mi)
                pairs.append(Pair(mi, ni))

        # Fallback: if no tail hit, try a broader suffix check against all missing rows
        # (still cheap at this data size, but we avoid it unless needed).
        if not matched_any:
            for ni, pname in df_missing["name_processed"].items():
                if suffix_match(network, pname):
                    master_to_missing[mi].add(ni)
                    missing_to_master[ni].add(mi)
                    pairs.append(Pair(mi, ni))

    df_master["n_missing_matches"] = df_master.index.map(lambda i: len(master_to_missing[i]))
    df_missing["n_master_matches"] = df_missing.index.map(lambda i: len(missing_to_master[i]))

    # Classify
    n_master_exact = int((df_master["n_missing_matches"] == 1).sum())
    n_missing_exact = int((df_missing["n_master_matches"] == 1).sum())
    # True one-to-one pairs are those where both sides have exactly one match and they match each other.
    one_to_one_pairs: List[Pair] = [
        p
        for p in pairs
        if df_master.at[p.master_idx, "n_missing_matches"] == 1
        and df_missing.at[p.missing_idx, "n_master_matches"] == 1
    ]

    one_to_one_pair_count = len({(p.master_idx, p.missing_idx) for p in one_to_one_pairs})

    one_to_many_master = int((df_master["n_missing_matches"] > 1).sum())
    many_to_one_missing = int((df_missing["n_master_matches"] > 1).sum())

    print("=== Suffix match: NetworkFilename vs processed name (name '__'->'/') ===")
    print(f"Master rows:  {len(df_master)}")
    print(f"Missing rows: {len(df_missing)}")
    print("")
    print("Counts:")
    print(f"- master rows w/ exactly 1 match: {n_master_exact}")
    print(f"- missing rows w/ exactly 1 match: {n_missing_exact}")
    print(f"- TRUE one-to-one matched pairs: {one_to_one_pair_count}")
    print(f"- one-to-many (master->many missing): {one_to_many_master}")
    print(f"- many-to-one (many master->missing): {many_to_one_missing}")
    print("")
    print("Unmatched:")
    print(f"- master rows w/ 0 matches: {(df_master['n_missing_matches'] == 0).sum()}")
    print(f"- missing rows w/ 0 matches: {(df_missing['n_master_matches'] == 0).sum()}")

    # Write outputs
    master_out = f"{OUT_DIR}/suffixmatch_master_matchcounts.csv"
    missing_out = f"{OUT_DIR}/suffixmatch_missing_matchcounts.csv"
    pairs_out = f"{OUT_DIR}/suffixmatch_pairs.csv"
    master_ambig_out = f"{OUT_DIR}/suffixmatch_master_ambiguous.csv"
    missing_ambig_out = f"{OUT_DIR}/suffixmatch_missing_ambiguous.csv"

    df_master.to_csv(master_out, index=False)
    df_missing.to_csv(missing_out, index=False)

    pd.DataFrame(
        [{
            "master_idx": p.master_idx,
            "missing_idx": p.missing_idx,
            "NetworkFilename": df_master.at[p.master_idx, "NetworkFilename"],
            "name": df_missing.at[p.missing_idx, "name"],
            "name_processed": df_missing.at[p.missing_idx, "name_processed"],
        } for p in pairs]
    ).drop_duplicates().to_csv(pairs_out, index=False)

    df_master[df_master["n_missing_matches"] != 1].to_csv(master_ambig_out, index=False)
    df_missing[df_missing["n_master_matches"] != 1].to_csv(missing_ambig_out, index=False)

    print("")
    print("Wrote:")
    print(f"- {master_out}")
    print(f"- {missing_out}")
    print(f"- {pairs_out}")
    print(f"- {master_ambig_out}")
    print(f"- {missing_ambig_out}")


if __name__ == "__main__":
    main()


