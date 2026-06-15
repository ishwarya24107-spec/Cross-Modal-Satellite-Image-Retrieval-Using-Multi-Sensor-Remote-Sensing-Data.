import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile as tffi

sar_path = r"C:\Users\ishwa\Downloads\archive\s1_validation\s1_validation\ROIs0000_validation_s1_0_p0.tif"
optical_path = r"C:\Users\ishwa\Downloads\archive\s2_validation\s2_validation\ROIs0000_validation_s2_0_p0.tif"

def visualize_pair(s1_file, s2_file):
    # 1. Read and fix the SAR image
    sar_img = tffi.imread(s1_file)
    
    # Check if shape is (Channels, H, W) vs (H, W, Channels)
    if sar_img.ndim == 3 and sar_img.shape[0] < sar_img.shape[1]:
        sar_band1 = sar_img[0, :, :]  # Grab first channel
    else:
        sar_band1 = sar_img[:, :, 0] if sar_img.ndim == 3 else sar_img
        
    sar_band1 = np.nan_to_num(sar_band1)
    sar_min, sar_max = np.percentile(sar_band1, (2, 98))
    sar_normalized = np.clip((sar_band1 - sar_min) / (sar_max - sar_min + 1e-5) * 255, 0, 255).astype(np.uint8)

    # 2. Read and fix the Optical RGB image
    opt_img = tffi.imread(s2_file)
    
    # Force the format to be (Height, Width, Channels)
    if opt_img.ndim == 3 and opt_img.shape[0] < opt_img.shape[1]:
        # If it's loaded as (Channels, H, W), transpose it to (H, W, Channels)
        opt_img = opt_img.transpose(1, 2, 0)
        
    # Grab the first 3 channels (True Color Red, Green, Blue)
    rgb_img = opt_img[:, :, :3].astype(np.float32)
    
    rgb_img = np.nan_to_num(rgb_img)
    opt_min, opt_max = np.percentile(rgb_img, (2, 98))
    rgb_normalized = np.clip((rgb_img - opt_min) / (opt_max - opt_min + 1e-5) * 255, 0, 255).astype(np.uint8)

    # 3. Plot them side-by-side with an explicit square bounding layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(sar_normalized, cmap='gray', aspect='equal')
    ax1.set_title("Sentinel-1 (Raw SAR Radar Patch)")
    ax1.axis('off')
    
    ax2.imshow(rgb_normalized, aspect='equal')
    ax2.set_title("Sentinel-2 (Optical RGB Patch)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if os.path.exists(sar_path) and os.path.exists(optical_path):
    print("Matrix dimensions adjusted. Launching square visualizer...")
    visualize_pair(sar_path, optical_path)
else:
    print("Error: Could not find files.")