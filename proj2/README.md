# Project 2: Gesture Recognition (HMM + K-means)

IMU-based gesture classification using K-means quantization and Hidden Markov Models (HMMs).

## Requirements

- Python 3
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Data

Place training (and optionally test) `.txt` files in a folder. By default the code uses:

- **Training:** `ECE5242Proj2-train/` (same directory as the scripts)
- **Testing:** `ECE5242Proj2-train/` (change `test_folder` in `main.py` for a different test set)

Each `.txt` file should have 7 columns (whitespace- or tab-separated): `Time`, `Wx`, `Wy`, `Wz`, `Ax`, `Ay`, `Az`. Filenames are used to infer gesture labels (e.g. `beat3_01.txt` → `beat3`, `circle31.txt` → `circle`).

## How to Run

Run from the **`proj2/`** directory (or set your working directory to `proj2/` so relative paths work).

### 1. Train K-means and build discrete sequences

```bash
cd proj2
python train_data.py
```

- Reads all `.txt` files from `ECE5242Proj2-train/`
- Fits K-means on 6D sensor data (Wx, Wy, Wz, Ax, Ay, Az)
- Converts each file to a sequence of cluster IDs and groups them by gesture
- **Outputs:** `kmeans_model.pkl`, `gesture_sequences.pkl`

### 2. Train HMMs per gesture

```bash
python hmm.py
```

- Loads `gesture_sequences.pkl` (must exist; run `train_data.py` first)
- Trains one left-to-right HMM per gesture with Baum–Welch
- **Outputs:** `hmm_models.pkl`

**Important:** In `hmm.py`, `n_observations` must equal `num_clusters` in `train_data.py` (e.g. 75 or 90). If you change one, change the other.

### 3. Run classification (test)

```bash
python main.py
```

- Loads `kmeans_model.pkl` and `hmm_models.pkl`
- For each `.txt` in the test folder: quantizes with K-means, scores with each HMM, prints top-3 gesture predictions by log-likelihood

To use a different test folder, edit `test_folder` in `main.py` (e.g. `ECE5242Proj2-test`).

### Optional: Plot IMU waveforms

```bash
python draw_gesture.py
```

- Plots all 6 sensor channels (Wx, Wy, Wz, Ax, Ay, Az) vs time for each `.txt` in `ECE5242Proj2-train/`. Change `folder_path` in the script to plot another folder.

## File overview

| File                  | Purpose |
|-----------------------|--------|
| `train_data.py`       | K-means training, quantization, gesture grouping → `kmeans_model.pkl`, `gesture_sequences.pkl` |
| `hmm.py`              | HMM training per gesture → `hmm_models.pkl` |
| `main.py`             | Load models, classify test files, print top-3 |
| `draw_gesture.py`     | Plot sensor waveforms from `.txt` files |

## Quick start (full pipeline)

```bash
cd proj2
python train_data.py
python hmm.py
python main.py
```
