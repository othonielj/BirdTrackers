#!/usr/bin/env python3
"""
02_detect_with_cvlib.py

This script performs bird detection using the cvlib library on preprocessed images.
cvlib is a wrapper around YOLO and other models that provides excellent general object detection.

The script:
1. Takes preprocessed images or raw images as input
2. Runs object detection to identify birds and other objects
3. Saves detection results to CSV and visualizes them
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bird detection using cvlib")
    parser.add_argument("--input_dir", type=str, default="data/input",
                        help="Directory containing image files (.jpg)")
    parser.add_argument("--output_dir", type=str, default="data/output",
                        help="Directory to save detection results")
    parser.add_argument("--confidence", type=float, default=0.1,
                        help="Confidence threshold for detection")
    parser.add_argument("--bird_only", action="store_true", 
                        help="Only keep bird detections")
    parser.add_argument("--model", type=str, default="yolov4", 
                        choices=["yolov3", "yolov4", "yolov3-tiny", "yolov4-tiny"],
                        help="Model to use for detection")
    return parser.parse_args()


def load_images(input_dir):
    """
    Load all jpg images from input directory, sorted by name.
    Returns a list of (frame_id, image_path) tuples.
    """
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_paths.sort()  # Ensure consistent ordering
    
    sequence = []
    for i, path in enumerate(image_paths):
        frame_id = i
        sequence.append((frame_id, path))
    
    return sequence


def run_detection(sequence, confidence_threshold=0.1, bird_only=False, model="yolov4"):
    """
    Run object detection on each image in the sequence.
    
    Args:
        sequence: List of (frame_id, image_path) tuples
        confidence_threshold: Detection confidence threshold
        bird_only: Only keep bird detections if True
        model: Model type for detection
        
    Returns:
        List of detection dictionaries
    """
    all_detections = []
    
    # Process each image
    for frame_id, image_path in tqdm(sequence, desc="Detecting objects"):
        # Load and preprocess image
        image = cv2.imread(image_path)
        
        # Run object detection
        boxes, labels, scores = cv.detect_common_objects(image, confidence=confidence_threshold, model=model)
        
        # Store detections
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            # Skip if not a bird and bird_only is True
            if bird_only and label != 'bird':
                continue
                
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            detection = {
                'frame': frame_id,
                'x': x1,
                'y': y1,
                'width': width,
                'height': height,
                'class_id': label,
                'score': score
            }
            
            all_detections.append(detection)
    
    return all_detections


def visualize_detections(sequence, detections, output_dir):
    """
    Visualize detections on the original images.
    
    Args:
        sequence: List of (frame_id, image_path) tuples
        detections: List of detection dictionaries
        output_dir: Directory to save visualization results
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, "detections_visualized")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Group detections by frame
    detections_by_frame = {}
    for detection in detections:
        frame = detection['frame']
        if frame not in detections_by_frame:
            detections_by_frame[frame] = []
        detections_by_frame[frame].append(detection)
    
    # Process each image
    for frame_id, image_path in tqdm(sequence, desc="Visualizing detections"):
        if frame_id not in detections_by_frame:
            continue
            
        # Load image
        image = cv2.imread(image_path)
        
        # Extract detections for this frame
        frame_detections = detections_by_frame[frame_id]
        
        # Prepare parameters for draw_bbox
        boxes = []
        labels = []
        scores = []
        
        for detection in frame_detections:
            x = detection['x']
            y = detection['y']
            w = detection['width']
            h = detection['height']
            boxes.append([x, y, x+w, y+h])
            labels.append(detection['class_id'])
            scores.append(detection['score'])
        
        # Draw bounding boxes
        output = draw_bbox(image, boxes, labels, scores)
        
        # Convert to RGB for saving with matplotlib
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        # Save visualization
        output_path = os.path.join(vis_dir, f"{frame_id:02d}_detection.jpg")
        plt.figure(figsize=(12, 8))
        plt.imshow(output_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    print(f"Visualizations saved to {vis_dir}")


def save_detections_to_csv(detections, output_path):
    """
    Save detection results to a CSV file.
    
    Args:
        detections: List of detection dictionaries
        output_path: Path to save the CSV file
    """
    if not detections:
        print("No detections to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(detections)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(detections)} detections to {output_path}")


def summarize_detections(detections):
    """
    Summarize detection results.
    
    Args:
        detections: List of detection dictionaries
    """
    if not detections:
        print("No detections found.")
        return
    
    # Count detections by class
    class_counts = {}
    for detection in detections:
        class_id = detection['class_id']
        if class_id not in class_counts:
            class_counts[class_id] = 0
        class_counts[class_id] += 1
    
    # Print summary
    print(f"Found {len(detections)} detections across {len(set(d['frame'] for d in detections))} frames.")
    print("Detections by class:")
    for class_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_id}: {count}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image sequence
    sequence = load_images(args.input_dir)
    if not sequence:
        print(f"No images found in {args.input_dir}")
        return
    
    print(f"Found {len(sequence)} images in sequence")
    
    # Run detection
    detections = run_detection(
        sequence, 
        confidence_threshold=args.confidence,
        bird_only=args.bird_only,
        model=args.model
    )
    
    # Summarize detections
    summarize_detections(detections)
    
    # Save detections to CSV
    csv_path = os.path.join(args.output_dir, "cvlib_detections.csv")
    save_detections_to_csv(detections, csv_path)
    
    # Visualize detections
    visualize_detections(sequence, detections, args.output_dir)
    
    print("Detection complete!")


if __name__ == "__main__":
    main() 