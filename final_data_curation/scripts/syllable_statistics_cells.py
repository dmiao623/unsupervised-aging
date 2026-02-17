from tqdm import tqdm

def _get_syllable_frequency_statistics(agg_results, th: float = 0.0):
    sequences = list(agg_results.values())
    uniq = sorted({s for seq in sequences for s in seq})
    
    if th > 0.0:
        global_counts = {s: 0 for s in uniq}
        for seq in sequences:
            for s in seq:
                global_counts[s] += 1
        total = sum(global_counts.values())
        uniq = [s for s in uniq if total and global_counts[s] / total >= th]

    if not uniq:
        return pd.DataFrame()

    idx = {s: i for i, s in enumerate(uniq)}
    n = len(uniq)

    out = {'name': []}
    for s in uniq:
        out[f"avg_bout_length_{s}"] = []
        out[f"total_duration_{s}"] = []
        out[f"num_bouts_{s}"] = []

    for name, seq in tqdm(agg_results.items(), desc="Processing sequences"):
        out['name'].append(name)
        
        total_len = [0]*n
        bout_cnt = [0]*n
        sum_dur = [0.0]*n

        prev = None
        run_len = 0
        for i, s in enumerate(seq):
            if s == prev:
                run_len += 1
            else:
                if prev in idx:
                    j = idx[prev]
                    total_len[j] += run_len
                    bout_cnt[j] += 1
                prev = s
                run_len = 1
            if s in idx:
                sum_dur[idx[s]] += 1.0

        if prev in idx:
            j = idx[prev]
            total_len[j] += run_len
            bout_cnt[j] += 1

        for j in range(n):
            sum_dur[j] = float(total_len[j])

        for j, s in enumerate(uniq):
            abl = (total_len[j] / bout_cnt[j]) if bout_cnt[j] else 0.0
            out[f"avg_bout_length_{s}"].append(abl)
            out[f"total_duration_{s}"].append(sum_dur[j])
            out[f"num_bouts_{s}"].append(int(bout_cnt[j]))

    df = pd.DataFrame(out)
    df.set_index('name', inplace=True)
    return df

syllable_stats_df = _get_syllable_frequency_statistics(agg_results)
print(f"Computed statistics for {len(syllable_stats_df)} sequences")
print(f"Columns: {len(syllable_stats_df.columns)} ({len(syllable_stats_df.columns)//3} unique syllables)")
syllable_stats_df.head()


def match_feature_df_to_stats(feature_df, syllable_stats_df, k=None):
    stat_cols = [col for col in syllable_stats_df.columns 
                 if col.startswith('avg_bout_length_') or 
                    col.startswith('total_duration_') or 
                    col.startswith('num_bouts_')]
    
    common_cols = [col for col in stat_cols if col in feature_df.columns]
    
    if k is not None:
        syllable_ids = sorted(set([
            int(col.split('_')[-1]) 
            for col in common_cols 
            if col.split('_')[-1].isdigit()
        ]))[:k]
        
        common_cols = [col for col in common_cols 
                      if col.split('_')[-1].isdigit() and 
                         int(col.split('_')[-1]) in syllable_ids]
    
    print(f"Found {len(common_cols)} common syllable statistic columns between feature_df and syllable_stats_df")
    print(f"Total columns in syllable_stats_df: {len(stat_cols)}")
    if k is not None:
        print(f"Using only first {k} syllables ({len(common_cols)} columns)")
    
    if len(common_cols) == 0:
        print("WARNING: No common syllable statistic columns found!")
        return 0, len(feature_df), []
    
    success_count = 0
    failure_count = 0
    match_details = []
    
    for idx, row in tqdm(feature_df.iterrows(), total=len(feature_df), desc="Matching rows"):
        network_filename = row.get('NetworkFilename', f'row_{idx}')
        feature_values = row[common_cols]
        
        matches = []
        for stats_idx, stats_row in syllable_stats_df.iterrows():
            stats_values = stats_row[common_cols]
            
            if all(abs(float(fv) - float(sv)) < 0.0001 if isinstance(fv, (int, float)) and isinstance(sv, (int, float)) 
                   else fv == sv 
                   for fv, sv in zip(feature_values, stats_values)):
                matches.append(stats_idx)
        
        if len(matches) == 1:
            success_count += 1
            match_details.append({
                'feature_row': idx,
                'network_filename': network_filename,
                'status': 'SUCCESS',
                'matched_to': matches[0],
                'num_matches': 1
            })
        else:
            failure_count += 1
            match_details.append({
                'feature_row': idx,
                'network_filename': network_filename,
                'status': 'FAILURE',
                'matched_to': matches if matches else None,
                'num_matches': len(matches)
            })
    
    return success_count, failure_count, match_details

success_count, failure_count, match_details = match_feature_df_to_stats(feature_df, syllable_stats_df, k=10)

print(f"\n{'='*60}")
print(f"MATCHING RESULTS:")
print(f"{'='*60}")
print(f"Total rows in feature_df: {len(feature_df)}")
print(f"SUCCESS (matched to exactly 1 element): {success_count}")
print(f"FAILURE (matched to 0 or >1 elements): {failure_count}")
print(f"Success rate: {100*success_count/(success_count+failure_count):.2f}%")
print(f"{'='*60}")

failures = [m for m in match_details if m['status'] == 'FAILURE']
if failures:
    print(f"\nFirst 5 failures:")
    for i, fail in enumerate(failures[:5]):
        print(f"  {i+1}. Row {fail['feature_row']} ({fail['network_filename']}): "
              f"{fail['num_matches']} matches")


def create_network_to_syllable_mapping(feature_df, agg_results, match_details):
    mapping_data = []
    
    for match in match_details:
        if match['status'] == 'SUCCESS':
            network_filename = match['network_filename']
            matched_name = match['matched_to']
            
            if matched_name in agg_results:
                syllable_sequence = agg_results[matched_name]
                mapping_data.append({
                    'NetworkFilename': network_filename,
                    'matched_name': matched_name,
                    'syllable_sequence': syllable_sequence
                })
    
    mapping_df = pd.DataFrame(mapping_data)
    return mapping_df

network_syllable_mapping = create_network_to_syllable_mapping(feature_df, agg_results, match_details)

print(f"\n{'='*60}")
print(f"NETWORK TO SYLLABLE MAPPING:")
print(f"{'='*60}")
print(f"Successfully mapped {len(network_syllable_mapping)} NetworkFilenames to syllable sequences")
print(f"{'='*60}")
network_syllable_mapping.head()
