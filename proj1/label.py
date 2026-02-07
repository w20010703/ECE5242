import numpy as np
from matplotlib import pyplot as plt
import cv2
from roipoly import RoiPoly
import os
folder = "data"
output_file = 'orange_pixels.npy'

for filename in os.listdir(folder):
    # Create image
    img = cv2.imread(os.path.join(folder,filename))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # Show the image
    fig = plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.title("left click: line segment         right click or double click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    roi1 = RoiPoly(color='r', fig=fig)

    # Show the image with both ROIs and their mean values
    plt.imshow(img)
    plt.colorbar()
    roi1.display_roi()
    roi1.display_mean(img[:,:,0])
    plt.title('The ROI')
    plt.show()

    mask = roi1.get_mask(img[:,:,0])
    orange_samples = img[mask]

    print(f"successfully labeled {orange_samples.shape[0]} orange pixels.")

    # save to file
    if os.path.exists(output_file):
        existing_data = np.load(output_file)
        combined_data = np.vstack((existing_data, orange_samples))
        np.save(output_file, combined_data)
        print(f"saved to {output_file} with {combined_data.shape[0]} samples.")
    else:
        np.save(output_file, orange_samples)
        print("saved to {output_file}.")