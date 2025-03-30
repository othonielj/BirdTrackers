#!/usr/bin/env python
"""
Compare Bird Tracking Results

This script compares the results of the original tracking method (B_track_birds.py)
and the deep learning enhanced tracker (deep_bird_tracker.py).

It provides:
1. Side-by-side visualizations of tracking results
2. Statistical comparison of tracking performance
3. Track continuity analysis
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import colorsys

def parse_args():
    parser = argparse.ArgumentParser(description="Compare bird tracking results")
    
    parser.add_argument("--original_csv", type=str, default="data/output/tracked_detections.csv",
                       help="Path to original tracking results CSV")
    parser.add_argument("--deep_csv", type=str, default="data/output/deep_tracked_detections.csv",
                       help="Path to deep tracker results CSV")
    parser.add_argument("--images_dir", type=str, default="data/input",
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="data/output/comparison",
                       help="Directory to save comparison results")
    parser.add_argument("--frames", type=str, default="0:500:50",
                       help="Range of frames to compare in format start:end:step")
    
    return parser.parse_args()

def load_tracks(csv_path):
    """Load tracking results from CSV file and organize by frame."""
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} does not exist")
        return {}
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Group by frame
    tracks_by_frame = {}
    for frame_id, group in df.groupby('frame'):
        tracks_by_frame[int(frame_id)] = group.to_dict('records')
    
    return tracks_by_frame

def load_images(input_dir):
    """Load image paths sorted by frame number."""
    # Get all image files
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
    
    # Sort images
    image_paths.sort()
    
    return image_paths

def analyze_tracking_stats(tracks_dict, name):
    """Analyze tracking statistics."""
    if not tracks_dict:
        return {"name": name, "total_frames": 0, "total_tracks": 0}
    
    # Collect all track IDs
    track_ids = set()
    for frame_id, tracks in tracks_dict.items():
        for track in tracks:
            track_ids.add(track['track_id'])
    
    # Count tracks per frame
    tracks_per_frame = [len(tracks) for frame_id, tracks in tracks_dict.items()]
    
    # Calculate average detections per track
    track_lengths = {}
    for frame_id, tracks in tracks_dict.items():
        for track in tracks:
            track_id = track['track_id']
            if track_id not in track_lengths:
                track_lengths[track_id] = 0
            track_lengths[track_id] += 1
    
    # Compute statistics
    stats = {
        "name": name,
        "total_frames": len(tracks_dict),
        "total_tracks": len(track_ids),
        "avg_tracks_per_frame": np.mean(tracks_per_frame) if tracks_per_frame else 0,
        "max_tracks_per_frame": max(tracks_per_frame) if tracks_per_frame else 0,
        "avg_track_length": np.mean(list(track_lengths.values())) if track_lengths else 0,
        "max_track_length": max(track_lengths.values()) if track_lengths else 0,
        "longest_track_id": max(track_lengths.items(), key=lambda x: x[1])[0] if track_lengths else None
    }
    
    return stats

def calculate_track_stability(tracks_dict):
    """Calculate track stability (how much tracks move between frames)."""
    if not tracks_dict:
        return {}
    
    # Track position changes
    track_movements = {}
    
    # Process frames in order
    frame_ids = sorted(tracks_dict.keys())
    
    for i in range(1, len(frame_ids)):
        prev_frame = frame_ids[i-1]
        curr_frame = frame_ids[i]
        
        # Get tracks from both frames
        prev_tracks = {t['track_id']: t for t in tracks_dict[prev_frame]}
        curr_tracks = {t['track_id']: t for t in tracks_dict[curr_frame]}
        
        # Find common tracks
        common_ids = set(prev_tracks.keys()) & set(curr_tracks.keys())
        
        # Calculate movement for each common track
        for track_id in common_ids:
            prev_track = prev_tracks[track_id]
            curr_track = curr_tracks[track_id]
            
            # Calculate center points
            prev_x = prev_track['x'] + prev_track['width']/2
            prev_y = prev_track['y'] + prev_track['height']/2
            curr_x = curr_track['x'] + curr_track['width']/2
            curr_y = curr_track['y'] + curr_track['height']/2
            
            # Calculate distance moved
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            # Store movement
            if track_id not in track_movements:
                track_movements[track_id] = []
            track_movements[track_id].append(distance)
    
    # Calculate average movement per track
    avg_movements = {}
    for track_id, movements in track_movements.items():
        avg_movements[track_id] = np.mean(movements)
    
    return avg_movements

def create_comparison_visualization(image, original_tracks, deep_tracks, frame_id):
    """Create a side-by-side comparison visualization."""
    # Create copy of the image for each tracker
    original_img = image.copy()
    deep_img = image.copy()
    
    # Create color maps for consistent colors
    original_colors = {}
    deep_colors = {}
    
    # Draw original tracks
    for track in original_tracks:
        track_id = track['track_id']
        
        # Assign color
        if track_id not in original_colors:
            hue = (track_id * 0.1) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR
            original_colors[track_id] = color
        
        color = original_colors[track_id]
        
        # Draw bounding box
        x, y = int(track['x']), int(track['y'])
        w, h = int(track['width']), int(track['height'])
        cv2.rectangle(original_img, (x, y), (x+w, y+h), color, 2)
        
        # Draw track ID
        cv2.putText(original_img, f"ID:{track_id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw deep tracks
    for track in deep_tracks:
        track_id = track['track_id']
        
        # Assign color
        if track_id not in deep_colors:
            hue = (track_id * 0.1) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR
            deep_colors[track_id] = color
        
        color = deep_colors[track_id]
        
        # Draw bounding box
        x, y = int(track['x']), int(track['y'])
        w, h = int(track['width']), int(track['height'])
        cv2.rectangle(deep_img, (x, y), (x+w, y+h), color, 2)
        
        # Draw track ID
        cv2.putText(deep_img, f"ID:{track_id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add titles
    title_bar_height = 40
    h, w = original_img.shape[:2]
    
    original_with_title = np.ones((h + title_bar_height, w, 3), dtype=np.uint8) * 255
    deep_with_title = np.ones((h + title_bar_height, w, 3), dtype=np.uint8) * 255
    
    original_with_title[title_bar_height:, :, :] = original_img
    deep_with_title[title_bar_height:, :, :] = deep_img
    
    # Add title text
    cv2.putText(original_with_title, f"Original Tracker (Frame {frame_id})", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(deep_with_title, f"Deep Learning Tracker (Frame {frame_id})", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Combine images horizontally
    comparison = np.hstack((original_with_title, deep_with_title))
    
    return comparison

def main():
    # Parse arguments
    args = parse_args()
    
    # Parse frame range
    frame_range = args.frames.split(":")
    if len(frame_range) == 3:
        start_frame = int(frame_range[0])
        end_frame = int(frame_range[1])
        step = int(frame_range[2])
    else:
        start_frame = 0
        end_frame = 500
        step = 50
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tracking results
    print("Loading tracking results...")
    original_tracks = load_tracks(args.original_csv)
    deep_tracks = load_tracks(args.deep_csv)
    
    # Load images
    print("Loading images...")
    image_paths = load_images(args.images_dir)
    
    # Generate comparison visualizations
    print("Generating visualizations...")
    for frame_id in tqdm(range(start_frame, min(end_frame, len(image_paths)), step)):
        # Load image
        image_path = image_paths[frame_id]
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        # Get tracks for this frame
        orig_frame_tracks = original_tracks.get(frame_id, [])
        deep_frame_tracks = deep_tracks.get(frame_id, [])
        
        # Create comparison visualization
        comparison = create_comparison_visualization(
            image, orig_frame_tracks, deep_frame_tracks, frame_id
        )
        
        # Save comparison
        output_path = os.path.join(args.output_dir, f"comparison_{frame_id:04d}.jpg")
        cv2.imwrite(output_path, comparison)
    
    # Analyze tracking statistics
    print("\nAnalyzing tracking statistics...")
    orig_stats = analyze_tracking_stats(original_tracks, "Original Tracker")
    deep_stats = analyze_tracking_stats(deep_tracks, "Deep Learning Tracker")
    
    # Print statistics
    print("\nTracking Statistics Comparison:")
    print(f"{'Metric':<25} {'Original':<15} {'Deep Learning':<15}")
    print("-" * 60)
    
    metrics = [
        ("Total Frames", "total_frames"),
        ("Total Tracks", "total_tracks"),
        ("Avg Tracks per Frame", "avg_tracks_per_frame"),
        ("Max Tracks per Frame", "max_tracks_per_frame"),
        ("Avg Track Length", "avg_track_length"),
        ("Max Track Length", "max_track_length"),
        ("Longest Track ID", "longest_track_id")
    ]
    
    for name, key in metrics:
        orig_val = orig_stats.get(key, "N/A")
        deep_val = deep_stats.get(key, "N/A")
        
        # Format numbers
        if isinstance(orig_val, (int, float)) and isinstance(deep_val, (int, float)):
            if isinstance(orig_val, int):
                print(f"{name:<25} {orig_val:<15d} {deep_val:<15d}")
            else:
                print(f"{name:<25} {orig_val:<15.2f} {deep_val:<15.2f}")
        else:
            print(f"{name:<25} {orig_val!s:<15} {deep_val!s:<15}")
    
    # Calculate track stability
    print("\nCalculating track stability...")
    orig_stability = calculate_track_stability(original_tracks)
    deep_stability = calculate_track_stability(deep_tracks)
    
    # Calculate average stability
    orig_avg_stability = np.mean(list(orig_stability.values())) if orig_stability else 0
    deep_avg_stability = np.mean(list(deep_stability.values())) if deep_stability else 0
    
    print(f"\nAverage track movement per frame:")
    print(f"Original Tracker: {orig_avg_stability:.2f} pixels")
    print(f"Deep Learning Tracker: {deep_avg_stability:.2f} pixels")
    
    # Conclude
    print(f"\nComparison visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 