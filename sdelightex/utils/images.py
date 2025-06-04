import cv2
import numpy as np
import argparse
import os
import re
import time

def extract_number(filename):
    """Extract numeric part from a filename for sorting purposes
    
    Args:
        filename (str): Input file name
    
    Returns:
        int: Extracted numeric value
    
    Raises:
        ValueError: If no numeric pattern is found in filename
    """
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No numeric pattern found in filename: {filename}")

def align_saturation(input_dir, reference_dir, output_dir, group_size=5, ext=("png", "jpg", "jpeg")):
    """
    Batch align saturation (S channel) of images in input directory to reference images.
    Processes images in groups and calculates optimal S-channel transfer parameters per group.
    
    Parameters:
        input_dir (str): Path to directory containing images to process
        reference_dir (str): Path to directory containing reference images
        output_dir (str): Output directory path for aligned images
        group_size (int): Number of images processed together to calculate S-channel parameters
        ext (tuple): Valid image file extensions (default: ('png', 'jpg', 'jpeg'))
    
    Returns:
        tuple: (processed_count, elapsed_time) 
        processed_count (int): Number of successfully processed images
        elapsed_time (float): Total processing time in seconds
    
    Raises:
        ValueError: If input and reference images don't match in quantity or naming
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize extensions to check
    valid_exts = tuple(f".{e.lower()}" for e in ext)
    
    # Gather and validate matching files in both directories
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    reference_files = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(valid_exts)])
    
    # Verify file lists match in length and content
    if len(input_files) != len(reference_files):
        raise ValueError("Input and reference image counts do not match")
    
    if any(inf != ref for inf, ref in zip(input_files, reference_files)):
        raise ValueError("Mismatched filenames between input and reference directories")
    
    # Create full path pairs sorted by numeric part of filename
    file_pairs = sorted(
        [(os.path.join(input_dir, f), os.path.join(reference_dir, f)) for f in input_files],
        key=lambda x: extract_number(x[0].split(os.sep)[-1])
    )
    
    # Divide image pairs into processing groups
    groups = [file_pairs[i:i+group_size] for i in range(0, len(file_pairs), group_size)]
    
    start_time = time.time()
    processed = 0
    
    # Process each group of images
    for group_idx, group in enumerate(groups):
        # Initialize accumulators for S-channel statistics
        sum_st, sum_s2, sum_s, sum_t, count = 0.0, 0.0, 0.0, 0.0, 0
        
        # Gather statistics from all image pairs in current group
        for inf_path, ref_path in group:
            img_input = cv2.imread(inf_path)
            img_ref = cv2.imread(ref_path)
            
            # Skip unreadable image pairs
            if img_input is None or img_ref is None:
                print(f"Warning: Could not read image {inf_path} or {ref_path}")
                continue
            
            # Convert to HSV color space and extract S channels
            s_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
            s_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
            
            # Compute absolute differences
            diff = np.abs(s_input - s_ref)
            
            # Identify pixels with smallest differences (most reliable matches)
            sorted_indices = np.argsort(diff.flatten())
            k = int(1.0 * diff.size)  # Select best 100% of pixels
            
            if k > 0:
                s_input_flat = s_input.flatten()
                s_ref_flat = s_ref.flatten()
                selected_indices = sorted_indices[:k]  # Indices of most similar pixels
                
                s_sel = s_input_flat[selected_indices]  # Input S values
                t_sel = s_ref_flat[selected_indices]   # Reference S values
                
                # Accumulate statistics for linear regression
                sum_st += np.sum(s_sel * t_sel)
                sum_s2 += np.sum(s_sel ** 2)
                sum_s += np.sum(s_sel)
                sum_t += np.sum(t_sel)
                count += k
        
        # Skip group if no valid pixels for calculation
        if count == 0:
            print(f"Warning: Group {group_idx} has no valid pixels for parameter calculation")
            continue
        
        # Compute linear transformation parameters (S_ref = alpha * S_input + beta)
        # Using ordinary least squares solution
        denominator = sum_s2 - (sum_s * sum_s) / count
        if denominator < 1e-6:  # Prevent division by zero
            alpha = 1.0
            beta = (sum_t - sum_s) / count
        else:
            alpha = (sum_st - sum_s * sum_t / count) / denominator
            beta = (sum_t - alpha * sum_s) / count
        
        # Apply calculated parameters to each image in the group
        for inf_path, ref_path in group:
            img = cv2.imread(inf_path)
            if img is None:
                continue
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Apply linear transformation to saturation channel
            s_transformed = alpha * s.astype(np.float32) + beta
            
            # Clip to valid range and convert back to uint8
            s_clipped = np.clip(s_transformed, 0, 255).astype(np.uint8)
            
            # Merge channels and convert back to BGR
            aligned = cv2.merge([h, s_clipped, v])
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_HSV2BGR)
            
            # Save result
            output_path = os.path.join(output_dir, os.path.basename(inf_path))
            cv2.imwrite(output_path, aligned_bgr)
            processed += 1
    
    # Report processing statistics
    total_time = time.time() - start_time
    print(f"Processing complete: {processed} images processed in {total_time:.2f} seconds")
    return processed, total_time
