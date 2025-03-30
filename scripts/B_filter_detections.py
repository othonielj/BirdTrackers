#!/usr/bin/env python3
"""
filter_detections.py

This script filters bird detections to remove false positives:
1. Filters detections based on minimum and maximum size
2. Excludes detections in the ground region
3. Applies non-maximum suppression to remove overlapping detections
4. Saves filtered detections to a new CSV file
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Filter bird detections")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to input CSV file with detections")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save filtered detections")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing original images for ground exclusion")
    parser.add_argument("--min_width", type=int, default=15,
                        help="Minimum width for bird detection (pixels)")
    parser.add_argument("--max_width", type=int, default=150,
                        help="Maximum width for bird detection (pixels)")
    parser.add_argument("--min_height", type=int, default=15,
                        help="Minimum height for bird detection (pixels)")
    parser.add_argument("--max_height", type=int, default=150,
                        help="Maximum height for bird detection (pixels)")
    parser.add_argument("--min_confidence", type=float, default=0.4,
                        help="Minimum confidence score")
    parser.add_argument("--nms_threshold", type=float, default=0.3,
                        help="IoU threshold for non-maximum suppression")
    parser.add_argument("--ground_exclusion_ratio", type=float, default=0.7,
                        help="Ratio of the image height from the bottom to exclude (0-1)")
    return parser.parse_args()


def filter_by_size_and_ratio(detections, min_width, max_width, min_height, max_height, min_confidence):
    """
    Filter detections based on size, aspect ratio, and confidence.
    
    Args:
        detections: DataFrame with detection data
        min_width: Minimum width in pixels
        max_width: Maximum width in pixels
        min_height: Minimum height in pixels
        max_height: Maximum height in pixels
        min_confidence: Minimum confidence score
    
    Returns:
        Filtered DataFrame
    """
    filtered = detections[
        (detections['width'] >= min_width) &
        (detections['width'] <= max_width) &
        (detections['height'] >= min_height) &
        (detections['height'] <= max_height) &
        (detections['score'] >= min_confidence)
    ].copy()
    
    # Calculate aspect ratio (width/height)
    filtered['aspect_ratio'] = filtered['width'] / filtered['height']
    
    # Filter out extremely elongated detections (not likely to be birds)
    filtered = filtered[
        (filtered['aspect_ratio'] >= 0.3) &
        (filtered['aspect_ratio'] <= 3.0)
    ]
    
    # Calculate area
    filtered['area'] = filtered['width'] * filtered['height']
    
    return filtered


def is_in_ground_region(detection, img_height, exclusion_ratio=0.7):
    """
    Check if a detection is in the ground region (lower part of the image).
    
    Args:
        detection: Dictionary with detection data
        img_height: Height of the image
        exclusion_ratio: Ratio of the image height from the bottom to exclude
    
    Returns:
        True if the detection is in the ground region, False otherwise
    """
    # Calculate the y-coordinate of the ground region boundary
    ground_boundary_y = img_height * (1 - exclusion_ratio)
    
    # Detection properties
    detection_bottom = detection['y'] + detection['height']
    detection_center_y = detection['y'] + detection['height'] / 2
    detection_area = detection['width'] * detection['height']
    
    # Criteria for identifying ground objects:
    
    # 1. Position-based checks
    is_completely_in_ground = detection['y'] > ground_boundary_y
    is_mostly_in_ground = detection_center_y > ground_boundary_y
    is_very_bottom = detection_bottom > img_height * 0.85
    
    # 2. Size and shape checks (ground objects tend to be larger with certain shapes)
    is_large_object = detection_area > (img_height * 0.05) * (img_height * 0.05)
    aspect_ratio = detection['width'] / max(detection['height'], 1)
    is_wide_object = aspect_ratio > 1.5 and detection['width'] > img_height * 0.1
    
    # 3. Combined checks for different types of ground objects
    
    # Large objects in the bottom portion
    is_large_ground_object = is_large_object and is_mostly_in_ground
    
    # Wide objects like sheep along the ground
    is_ground_animal = is_wide_object and is_very_bottom
    
    # Completely in ground region
    is_definite_ground = is_completely_in_ground and detection_area > (img_height * 0.02) * (img_height * 0.02)
    
    # Return true if any ground object criteria are met
    return is_large_ground_object or is_ground_animal or is_definite_ground


def filter_ground_objects(detections, images_dir, exclusion_ratio=0.7):
    """
    Filter out detections in the ground region.
    
    Args:
        detections: DataFrame with detection data
        images_dir: Directory containing original images
        exclusion_ratio: Ratio of the image height from the bottom to exclude
    
    Returns:
        Filtered DataFrame
    """
    # Load any image to get dimensions
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    if not image_files:
        print("Warning: No images found in images_dir, skipping ground exclusion")
        return detections
        
    sample_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    img_height, img_width = sample_image.shape[:2]
    
    # Create a boolean mask for detections to keep
    mask = []
    for _, row in detections.iterrows():
        detection = row.to_dict()
        mask.append(not is_in_ground_region(detection, img_height, exclusion_ratio))
    
    # Use boolean mask to filter detections
    return detections[mask]


def apply_nms(detections, iou_threshold=0.3):
    """
    Apply non-maximum suppression to detections per frame.
    
    Args:
        detections: DataFrame with detection data
        iou_threshold: IoU threshold for NMS
    
    Returns:
        DataFrame with filtered detections
    """
    result_dfs = []
    
    # Group by frame
    for frame, frame_dets in detections.groupby('frame'):
        if len(frame_dets) <= 1:
            result_dfs.append(frame_dets)
            continue
            
        # Convert to format expected by NMS
        boxes = []
        for _, row in frame_dets.iterrows():
            x1 = row['x']
            y1 = row['y']
            x2 = x1 + row['width']
            y2 = y1 + row['height']
            boxes.append([x1, y1, x2, y2, row['score']])
            
        boxes = np.array(boxes)
        
        # Get indices of boxes to keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        # Keep only selected boxes
        result_dfs.append(frame_dets.iloc[keep])
    
    return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()


def main():
    args = parse_args()
    
    # Load detections
    print(f"Loading detections from {args.input_csv}")
    detections = pd.read_csv(args.input_csv)
    initial_count = len(detections)
    print(f"Loaded {initial_count} detections")
    
    # Filter by size, aspect ratio, and confidence
    print("Filtering by size, aspect ratio, and confidence...")
    filtered = filter_by_size_and_ratio(
        detections,
        args.min_width,
        args.max_width,
        args.min_height,
        args.max_height,
        args.min_confidence
    )
    size_filtered_count = len(filtered)
    print(f"After size filtering: {size_filtered_count} detections ({initial_count - size_filtered_count} removed)")
    
    # Filter out ground objects
    print("Filtering out ground objects...")
    filtered = filter_ground_objects(
        filtered,
        args.images_dir,
        args.ground_exclusion_ratio
    )
    ground_filtered_count = len(filtered)
    print(f"After ground filtering: {ground_filtered_count} detections ({size_filtered_count - ground_filtered_count} removed)")
    
    # Apply NMS
    print("Applying non-maximum suppression...")
    final_detections = apply_nms(filtered, args.nms_threshold)
    final_count = len(final_detections)
    print(f"After NMS: {final_count} detections ({ground_filtered_count - final_count} removed)")
    
    # Remove temporary columns before saving
    if 'aspect_ratio' in final_detections.columns:
        final_detections = final_detections.drop(columns=['aspect_ratio'])
    if 'area' in final_detections.columns:
        final_detections = final_detections.drop(columns=['area'])
    
    # Save filtered detections
    final_detections.to_csv(args.output_csv, index=False)
    print(f"Saved {final_count} filtered detections to {args.output_csv}")
    print(f"Removed {initial_count - final_count} detections in total ({(initial_count - final_count) / initial_count * 100:.1f}%)")


if __name__ == "__main__":
    main() 