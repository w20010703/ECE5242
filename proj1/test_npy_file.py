import numpy as np

# Load file
data = np.load('orange_pixels.npy')

print(f"shape: {data.shape}")
print(f"type: {data.dtype}")
print(f"data:\n{data[:5]}")