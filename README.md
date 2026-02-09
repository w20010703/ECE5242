# ECE5242

## Project 1: Orange cone detection and distance estimation

This repository contains code for segmenting orange traffic cones in images and estimating their distance using a GMM-based classifier and simple geometry.

---

## Instructions: how to run on test images

### 1. Required files in `proj1/`

Ensure these are present in the **`proj1/`** directory:

- `main.py` — main script that runs the algorithm on all images in a folder
- `trained_gmm_k20.npz` — pre-trained GMM model (required; do not remove)

### 2. Test image folder

- By default, the script reads images from the **`test`** folder inside `proj1/`.
- Place your test images (e.g. `.png`, `.jpg`) in **`proj1/test/`**.
- To use a different folder: open `main.py` and change the line  
  `folder = "test"`  
  to your folder name (e.g. `folder = "grader_test"`). The path is relative to `proj1/`.

### 3. Running the code

From the **repository root** (`ECE5242/`):

```bash
cd proj1
python main.py
```

Or from any directory, by making `proj1` the current working directory so that relative paths resolve correctly:

```bash
python /path/to/ECE5242/proj1/main.py
```

(If you run from elsewhere, ensure your shell’s current working directory is `proj1/`, e.g. `cd proj1` first.)

### 4. During execution

- The script processes **one image at a time**.
- Two windows open: **Segmented Mask** and **Detection Result** (bounding box, base point, distance).
- Press **any key** (e.g. Space) to advance to the next image. Press **`q`** to skip to the next image without closing.
- Order of images follows `os.listdir(folder)` (not necessarily alphabetical).

### 5. Output

- **Console:** For each image, a line is printed, e.g.  
  `Image:train_1_dist915.png, Down:432.00, Right:314.00, Dist:985.08`
- **File:** The same lines are **appended** to **`proj1/wc683_results.txt`**.  
  Before each run, the script **clears** this file, so one run produces one results file.
- **Format per line:**  
  `Image:<filename>, Down:<y>, Right:<x>, Dist:<distance_cm>`
  - **Down:** row index of cone base (bottom).
  - **Right:** column index of cone center (horizontal).
  - **Dist:** estimated distance in cm.

No training or extra setup is required; the provided `trained_gmm_k20.npz` is used as-is.
