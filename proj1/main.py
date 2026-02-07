import cv2
import os
import numpy as np
from skimage.measure import label, regionprops

# --- 1. Load trained model parameters ---
model_data = np.load('trained_gmm_k20.npz')
PI = model_data['pi']
MU = model_data['mu']
SIGMA = model_data['sigma']
K = len(PI)
print("Model orange centers (MU):")
print(MU)

def gaussian_pdf(x, mu, sigma):
    d = x.shape[1]
    # Adding 1e-2 to sigma makes the model more permissive for color matching
    sigma_fixed = sigma + np.eye(d) * 1e-2 
    det = np.linalg.det(sigma_fixed)
    inv = np.linalg.inv(sigma_fixed)
    
    # Prevent norm_coeff from becoming inf when det is too small
    norm_coeff = 1.0 / (np.power((2 * np.pi), d/2) * np.sqrt(det) + 1e-9)
    
    diff = x - mu
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    
    # Clip exponent to avoid exp underflow/overflow
    exponent = np.clip(exponent, -50, 50) 
    
    return norm_coeff * np.exp(exponent)

def myAlgorithm(img):
    # --- Step A: Color conversion (must match training; here YCrCb) ---
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    h, w, _ = img_hsv.shape
    pixels = img_hsv.reshape(-1, 3).astype(np.float64)

    # --- Step B: Compute GMM scores (pixel-wise classification) ---
    scores = np.zeros(len(pixels))

    for k in range(K):
        scores += PI[k] * gaussian_pdf(pixels, MU[k], SIGMA[k])
    
    # --- Step C: Produce mask (adjust threshold based on your results) ---
    # You can print np.max(scores) to tune the threshold
    # print(f"Scores Max: {np.max(scores)}")
    thresh = np.max(scores) * 0.01
    # print(f"Current Threshold: {thresh}")
    mask = (scores > thresh).reshape(h, w).astype(np.uint8) * 255

    # --- Image processing (morphology) ---
    kernel = np.ones((10, 10), np.uint8) 
    # Close: dilate then erode, fills holes/gaps in the cone
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Open: erode then dilate, removes small background noise (optional)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # -----------------------------------

    # --- Step D: Localization ---
    label_img = label(mask)
    props = regionprops(label_img)

    if not props:
        return 0, 0, 0, mask

    # 1. Set filter thresholds
    min_area = 100  # Filter out tiny noise
    candidates = []

    for p in props:
        if p.area < min_area:
            continue
        minr, minc, maxr, maxc = p.bbox
        height = maxr - minr
        width = maxc - minc
        if width == 0:
            continue
        aspect_ratio = height / width
        # print(f"height: {height}, width: {width}")
        # print(f"aspect_ratio: {aspect_ratio}")

        # 2. Check aspect ratio for cone-like shape (roughly 2–4)
        if 2.0 < aspect_ratio < 3.0:
            candidates.append(p)
    # print(f"candidates: {len(candidates)}")
    # 3. If any candidates match shape, pick the one with largest area
    if candidates:
        main_cone = max(candidates, key=lambda x: x.area)
    else:
        # If none match aspect ratio, fallback to largest area (or return 0)
        main_cone = max(props, key=lambda x: x.area)

    # (Alternative: take largest region as cone)
    # main_cone = max(props, key=lambda x: x.area)
    minr, minc, maxr, maxc = main_cone.bbox
    
    down = float(maxr)   # Bottom row
    right = (minc + maxc) / 2.0  # Center column
    
    # --- Step E: Distance estimation ---
    # Focal length f (use your calibrated value)
    real_h = 43.18
    pixel_h = maxr - minr
    f = -0.22 * pixel_h + 691
    dist = (real_h * f) / (pixel_h)

    

    return down, right, dist, mask, main_cone

# --- Main loop ---
folder = "test"  # Your folder path
for filename in os.listdir(folder):
    # if not filename.endswith(('.png', '.jpg', '.jpeg')): continue
    
    img = cv2.imread(os.path.join(folder, filename))
    if img is None: continue

    # Run algorithm
    y, x, d, mask, main_cone = myAlgorithm(img)

    # Visualize results
    result_img = img.copy()
    if d > 0:
        # Draw bounding box
        # label_img = label(mask)
        # main_cone = max(regionprops(label_img), key=lambda x: x.area)
        minr, minc, maxr, maxc = main_cone.bbox
        cv2.rectangle(result_img, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
        print(f"height: {maxr - minr}, dist: {d}, filename: {filename}")
        # Draw base point
        cv2.circle(result_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        # Show distance
        cv2.putText(result_img, f"{d:.2f}cm", (minc, minr-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # print(f"Image:{filename}, Down:{y:.2f}, Right:{x:.2f}, Dist:{d:.2f}")
    
    # Show results
    cv2.imshow('Segmented Mask', mask)
    cv2.imshow('Detection Result', result_img)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue

cv2.destroyAllWindows()