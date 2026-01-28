`2025-03-10_b6-do_frailty.csv`: 
- taken from https://github.com/KumarLabJax/DO-vFI-modeling/blob/main/Data/B6DO_frailty_video.csv
`2025-11-03_missing-paths.txt`: 
- taken from https://thejacksonlaboratory.slack.com/archives/D08VB8UN6Q5/p1763046939447309
- contains pose files missing from `2025-12-03_deduped-ds.csv`
`2025-11-03_missing-paths_parsed.txt`:
- parsed paths from `2025-11-03_missing-paths.txt` for matching in `2025-03-10_b6-do_frailty.csv`
`2025-12-03_deduped-ds.csv`:
- taken from https://jacksonlaboratory.enterprise.slack.com/files/U0900NN9DAQ/F09Q4VD2N9W/deduped_dataset.csv
- contains metadata, supervised features, and unsupervised feature (combined model)
`2025-12-03_missing-paths_corrected.txt`:
- entries from `2025-11-03_missing-paths.txt` that have corresponding metadata and are not already present
`2025-12-04_final-name-registry.txt`
- names of all finalized experiment names, combined from `2025-12-03_missing-paths_corrected.txt` and `2025-12-03_deduped-ds`