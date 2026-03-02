import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pickle
import re

# Training file folder (path relative to this script's directory)
folder_path = 'ECE5242Proj2-train'
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_abs = os.path.join(script_dir, folder_path)
training_files = []
for file in sorted(os.listdir(folder_abs)):
    if file.endswith('.txt'):
        training_files.append(os.path.join(folder_abs, file))
if not training_files:
    raise SystemExit(f"No .txt files found in {folder_abs}")
columns = ['Time', 'Wx', 'Wy', 'Wz', 'Ax', 'Ay', 'Az']

def load_sensor_data(filepath):
    """Load one file; return sensor matrix (Nx6) without Time column."""
    df = pd.read_csv(
        filepath,
        header=None,
        sep=r'\s+',
        engine='python',
        skip_blank_lines=True,
    )
    if df.shape[1] != 7:
        raise ValueError(f"Expected 7 columns in {filepath}, got {df.shape[1]}")
    df.columns = columns
    return df[['Wx', 'Wy', 'Wz', 'Ax', 'Ay', 'Az']].values

print("Loading data for K-means training...")
all_data_list = []
for f in training_files:
    data = load_sensor_data(f)
    all_data_list.append(data)

all_data_stacked = np.vstack(all_data_list)
print(f"Total data shape for K-means: {all_data_stacked.shape}")

num_clusters = 75
print(f"Training K-means with {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(all_data_stacked)

# Save K-means model to script directory (use same model at test time)
model_path = os.path.join(script_dir, 'kmeans_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(kmeans, f)
print("K-means model saved.")

discrete_sequences = []

for f in training_files:
    data = load_sensor_data(f)
    discrete_seq = kmeans.predict(data) 
    discrete_sequences.append(discrete_seq)
    print(f"File {os.path.basename(f)} converted to discrete sequence of length {len(discrete_seq)}")

# Draw discrete sequence (first file) as a graph
# seq = discrete_sequences[0]
# plt.figure(figsize=(12, 4))
# plt.plot(np.arange(len(seq)), seq, drawstyle='steps-post', color='steelblue')
# plt.xlabel('Time step')
# plt.ylabel('Cluster ID')
# plt.title(f'Discrete sequence: {os.path.basename(training_files[0])}')
# plt.grid(True)
# plt.tight_layout()
# plt.show()



gesture_sequences = {}

for f, seq in zip(training_files, discrete_sequences):
    
    filename = os.path.basename(f).lower()
    if 'beat3' in filename:
        gesture_name = 'beat3'
    elif 'beat4' in filename:
        gesture_name = 'beat4'
    elif 'inf' in filename:
        gesture_name = 'infinity'
    elif 'wave' in filename:
        gesture_name = 'wave'
    elif 'circle' in filename:
        gesture_name = 'circle'
    elif 'eight' in filename:
        gesture_name = 'eight'
    else:
        continue

    if gesture_name not in gesture_sequences:
        gesture_sequences[gesture_name] = []
    gesture_sequences[gesture_name].append(seq)

seq_model_path = os.path.join(script_dir, 'gesture_sequences.pkl')
with open(seq_model_path, 'wb') as f:
    pickle.dump(gesture_sequences, f)

print("Discrete sequences grouped by gesture and saved successfully!")
for g, seqs in gesture_sequences.items():
    print(f"  - {g}: {len(seqs)} sequences")