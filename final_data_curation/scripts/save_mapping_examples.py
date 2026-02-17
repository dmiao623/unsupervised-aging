import pandas as pd
import numpy as np
import pickle

# Option 1: Parquet (RECOMMENDED - efficient, preserves types, widely supported)
network_syllable_mapping.to_parquet('network_syllable_mapping.parquet', index=False)
loaded_df = pd.read_parquet('network_syllable_mapping.parquet')

# Option 2: Pickle (Simple, preserves exact Python objects)
with open('network_syllable_mapping.pkl', 'wb') as f:
    pickle.dump(network_syllable_mapping, f)

with open('network_syllable_mapping.pkl', 'rb') as f:
    loaded_df = pickle.load(f)

# Option 3: HDF5 (Good if already using h5 files in your pipeline)
network_syllable_mapping.to_hdf('network_syllable_mapping.h5', key='mapping', mode='w')
loaded_df = pd.read_hdf('network_syllable_mapping.h5', key='mapping')

# Option 4: CSV with sequences as strings (if you need human-readable format)
mapping_for_csv = network_syllable_mapping.copy()
mapping_for_csv['syllable_sequence'] = mapping_for_csv['syllable_sequence'].apply(
    lambda x: ','.join(map(str, x))
)
mapping_for_csv.to_csv('network_syllable_mapping.csv', index=False)

loaded_csv = pd.read_csv('network_syllable_mapping.csv')
loaded_csv['syllable_sequence'] = loaded_csv['syllable_sequence'].apply(
    lambda x: np.array([int(i) for i in x.split(',')])
)

# Option 5: NumPy npz (if you only care about the sequences, not the full dataframe)
np.savez('network_syllable_mapping.npz',
         network_filenames=network_syllable_mapping['NetworkFilename'].values,
         matched_names=network_syllable_mapping['matched_name'].values,
         syllable_sequences=np.array(network_syllable_mapping['syllable_sequence'].tolist(), dtype=object))

loaded = np.load('network_syllable_mapping.npz', allow_pickle=True)
