#!/usr/bin/env python3
"""
B_track_birds.py

This script tracks birds across multiple frames:
1. Loads detection results from a CSV file
2. Associates detections across frames using spatial and appearance features
3. Saves tracking results with track IDs to a CSV file
4. Creates visualizations of the tracking results
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
import random
from filterpy.kalman import KalmanFilter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Track birds across frames")
    parser.add_argument("--detections_csv", type=str, required=True,
                        help="Path to CSV file with detections")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing original images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save tracking results")
    parser.add_argument("--max_distance", type=float, default=150,
                        help="Maximum distance for matching tracks to detections")
    parser.add_argument("--iou_threshold", type=float, default=0.3,
                        help="IoU threshold for matching")
    parser.add_argument("--inactive_frames", type=int, default=30,
                        help="Maximum number of frames a track can be inactive")
    parser.add_argument("--appearance_weight", type=float, default=0.3,
                        help="Weight of appearance features in matching (0-1)")
    parser.add_argument("--min_hits", type=int, default=3,
                        help="Minimum number of detections for a confirmed track")
    parser.add_argument("--history_frames", type=int, default=20,
                        help="Number of previous frames to show in trajectory")
    return parser.parse_args()


def load_images(input_dir):
    """
    Load all jpg images from input directory, sorted by name.
    Returns a list of (frame_id, image_path) tuples.
    """
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_paths.sort()
    
    sequence = []
    for i, path in enumerate(image_paths):
        frame_id = i
        sequence.append((frame_id, path))
    
    return sequence


def load_detections(csv_path):
    """
    Load bird detections from CSV file.
    Returns a list of detection dictionaries grouped by frame.
    """
    # Read CSV file
    detections_df = pd.read_csv(csv_path)
    
    # Group by frame
    detections_by_frame = {}
    for frame_id, group in detections_df.groupby('frame'):
        detections_by_frame[frame_id] = []
        for _, row in group.iterrows():
            detection = {
                'frame': row['frame'],
                'x': row['x'],
                'y': row['y'],
                'width': row['width'],
                'height': row['height'],
                'score': row['score'],
                'class_id': row.get('class_id', 'bird')
            }
            detections_by_frame[frame_id].append(detection)
    
    return detections_by_frame


class BirdTrack:
    """Track class for bird tracking."""
    
    def __init__(self, detection, track_id):
        self.id = track_id
        self.detections = [detection]
        self.frames = [detection['frame']]
        self.inactive_count = 0
        self.active = True
        self.confirmed = False
        self.hit_count = 1
        
        # Calculate initial location and size
        self.x = detection['x']
        self.y = detection['y']
        self.width = detection['width']
        self.height = detection['height']
        
        # Initialize velocity
        self.vx = 0
        self.vy = 0
        
        # Initialize appearance features
        self.appearance = None
        self.appearance_history = []  # Store multiple appearance samples
        self.max_appearance_history = 5  # Maximum number of appearance samples to store
        
        # Initialize Kalman filter
        self.initialize_kalman_filter()
    
    def initialize_kalman_filter(self):
        """Initialize enhanced Kalman filter for bird flight tracking."""
        # 6D state: [x, y, vx, vy, ax, ay] - position, velocity, acceleration
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0  # time step
        
        # State transition matrix - constant acceleration model
        self.kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe x,y)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Set initial state - center of bounding box
        center_x = self.x + self.width/2
        center_y = self.y + self.height/2
        self.kf.x = np.array([[center_x], [center_y], [0], [0], [0], [0]])
                              
        # Measurement uncertainty
        self.kf.R = np.eye(2) * 5
        
        # Process uncertainty
        self.kf.Q = np.eye(6) * 0.1
        self.kf.Q[2:4, 2:4] *= 5  # Higher uncertainty for velocity
        self.kf.Q[4:6, 4:6] *= 10  # Even higher uncertainty for acceleration
        
        # Initial covariance
        self.kf.P = np.eye(6) * 100
        
        # For adaptive noise
        self.innovation_history = []
        self.max_innovation_history = 5
    
    def predict(self):
        """Predict next position with enhanced Kalman filter."""
        # Adjust process noise based on recent innovations if we have history
        if len(self.innovation_history) >= 3:
            self.adapt_process_noise()
        
        # Predict next state with Kalman filter
        self.kf.predict()
        
        # Extract predicted center coordinates
        center_x = self.kf.x[0, 0]
        center_y = self.kf.x[1, 0]
        
        # Update velocity and acceleration from Kalman filter
        self.vx = self.kf.x[2, 0]
        self.vy = self.kf.x[3, 0]
        
        # Calculate predicted bounding box position from center
        predicted_x = center_x - self.width/2
        predicted_y = center_y - self.height/2
        
        # Update position based on Kalman prediction
        self.x = predicted_x
        self.y = predicted_y
        
        # Apply physical constraints to predictions
        self.apply_flight_constraints()
        
        return self.x, self.y, self.width, self.height
    
    def apply_flight_constraints(self):
        """Apply physical constraints typical of bird flight patterns."""
        # Maximum reasonable speed for a bird in pixels per frame
        max_speed = 50
        speed = np.sqrt(self.vx**2 + self.vy**2)
        
        # If speed is too high, scale it down
        if speed > max_speed:
            scale_factor = max_speed / speed
            self.vx *= scale_factor
            self.vy *= scale_factor
            
            # Update Kalman state with constrained velocity
            self.kf.x[2, 0] = self.vx
            self.kf.x[3, 0] = self.vy
    
    def adapt_process_noise(self):
        """Adapt Kalman filter process noise based on recent innovations."""
        # Calculate average innovation magnitude
        avg_innovation = np.mean([np.linalg.norm(inn) for inn in self.innovation_history])
        
        # Scale factor based on innovation (higher innovation = higher process noise)
        scale = max(1.0, min(3.0, avg_innovation / 10.0))
        
        # Adapt process noise for velocity and acceleration
        base_velocity_noise = 0.1 * 5
        base_accel_noise = 0.1 * 10
        
        # Update Q matrix diagonal elements for velocity and acceleration
        self.kf.Q[2, 2] = self.kf.Q[3, 3] = base_velocity_noise * scale
        self.kf.Q[4, 4] = self.kf.Q[5, 5] = base_accel_noise * scale
    
    def update(self, detection, image=None):
        """Update track with new detection."""
        prev_x, prev_y = self.x, self.y
        
        # Update position and size
        self.x = detection['x']
        self.y = detection['y']
        self.width = detection['width']
        self.height = detection['height']
        
        # Calculate center for Kalman update
        center_x = self.x + self.width/2
        center_y = self.y + self.height/2
        
        # Store pre-update state for innovation calculation
        pre_update_x = self.kf.x[0, 0]
        pre_update_y = self.kf.x[1, 0]
        
        # Update Kalman filter with new measurement
        measurement = np.array([[center_x], [center_y]])
        self.kf.update(measurement)
        
        # Calculate innovation (difference between prediction and measurement)
        innovation = np.array([center_x - pre_update_x, center_y - pre_update_y])
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.max_innovation_history:
            self.innovation_history.pop(0)
        
        # Update velocity based on Kalman filter
        self.vx = self.kf.x[2, 0]
        self.vy = self.kf.x[3, 0]
        
        # Add detection to track
        self.detections.append(detection)
        self.frames.append(detection['frame'])
        self.inactive_count = 0
        self.hit_count += 1
        
        # Update confirmed status
        if self.hit_count >= 3 and not self.confirmed:
            self.confirmed = True
        
        # Update appearance features if image is provided
        if image is not None:
            self.update_appearance(image)
    
    def update_appearance(self, image):
        """Extract and update appearance features from the image."""
        # Extract the bird region
        x, y, w, h = int(self.x), int(self.y), int(self.width), int(self.height)
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Extract the patch
        bird_patch = image[y:y+h, x:x+w]
        
        if bird_patch.size > 0:
            # Extract enhanced features
            new_appearance = self.extract_enhanced_features(bird_patch)
            
            # Set current appearance
            self.appearance = new_appearance
            
            # Add to appearance history
            self.appearance_history.append(new_appearance)
            if len(self.appearance_history) > self.max_appearance_history:
                self.appearance_history.pop(0)
    
    def extract_enhanced_features(self, image):
        """Extract enhanced appearance features from image patch."""
        # Resize image for consistent feature extraction
        target_size = (32, 32)
        if image.shape[0] > 0 and image.shape[1] > 0:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        else:
            return np.zeros(32*3 + 32*32*3)  # Return empty features if image is invalid
        
        # 1. Color histograms (HSV space)
        try:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Normalize histograms
            hist_h = cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX).flatten()
            hist_s = cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX).flatten()
            hist_v = cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX).flatten()
        except Exception as e:
            # If histogram calculation fails, create empty histograms
            print(f"Warning: Error calculating histograms: {e}")
            hist_h = np.zeros(32)
            hist_s = np.zeros(32)
            hist_v = np.zeros(32)
        
        # 2. Pixel intensity values (flattened and normalized)
        normalized_pixels = resized.flatten() / 255.0
        
        # Combine features
        features = np.concatenate([hist_h, hist_s, hist_v, normalized_pixels])
        return features
    
    def compare_appearance(self, other_track):
        """Compare appearance features with another track using advanced matching."""
        if self.appearance is None or other_track.appearance is None:
            return 0.0
        
        # If we have appearance history, use the best match
        if self.appearance_history and other_track.appearance_history:
            similarities = []
            
            # Compare current appearance with all histories
            for hist_appearance in other_track.appearance_history:
                # Extract histogram portions
                self_hist = self.appearance[:96].reshape(-1, 1).astype(np.float32)
                other_hist = hist_appearance[:96].reshape(-1, 1).astype(np.float32)
                
                # Use both histogram correlation and L2 distance
                hist_corr = cv2.compareHist(self_hist, other_hist, cv2.HISTCMP_CORREL)
                
                # Extract normalized pixel intensities
                self_pixels = self.appearance[96:]
                other_pixels = hist_appearance[96:]
                
                # Calculate L2 distance and convert to similarity
                l2_dist = np.sqrt(np.sum((self_pixels - other_pixels) ** 2))
                pixel_similarity = np.exp(-0.5 * (l2_dist / 10.0))  # Convert distance to similarity
                
                # Weighted combination
                similarity = 0.7 * hist_corr + 0.3 * pixel_similarity
                similarities.append(similarity)
            
            # Return best match
            if similarities:
                return max(max(0.0, sim) for sim in similarities)
        
        # Fallback to simple histogram correlation
        try:
            self_hist = self.appearance[:96].reshape(-1, 1).astype(np.float32)
            other_hist = other_track.appearance[:96].reshape(-1, 1).astype(np.float32)
            correlation = cv2.compareHist(self_hist, other_hist, cv2.HISTCMP_CORREL)
            return max(0.0, correlation)  # Ensure non-negative
        except Exception as e:
            # If any error occurs, return a default similarity
            print(f"Warning: Error comparing histograms: {e}")
            return 0.5  # Default moderate similarity
    
    def mark_inactive(self):
        """Mark track as inactive but continue prediction."""
        self.inactive_count += 1
        
        # Still perform Kalman prediction even when inactive
        self.predict()
        
        if self.inactive_count > 30:  # Maximum inactive frames
            self.active = False
    
    def get_last_detection(self):
        """Get the most recent detection."""
        return self.detections[-1]
    
    def get_last_position(self):
        """Get the most recent position."""
        return self.x, self.y, self.width, self.height
    
    def get_predicted_position(self):
        """Get the predicted next position."""
        center_x = self.kf.x[0, 0]
        center_y = self.kf.x[1, 0]
        return center_x - self.width/2, center_y - self.height/2, self.width, self.height
    
    def get_center(self):
        """Get the center point of the bounding box."""
        return self.x + self.width / 2, self.y + self.height / 2


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    # Extract coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / max(union_area, 1e-6)
    return iou


def calculate_cost_matrix(tracks, detections, frame_image=None, max_distance=150, appearance_weight=0.3):
    """
    Calculate cost matrix for matching tracks to detections.
    Uses a combination of spatial distance, motion prediction, and appearance similarity.
    """
    n_tracks = len(tracks)
    n_detections = len(detections)
    
    if n_tracks == 0 or n_detections == 0:
        return np.zeros((n_tracks, n_detections))
    
    # Initialize cost matrix with large values
    cost_matrix = np.full((n_tracks, n_detections), float('inf'))
    
    # Calculate costs for each track-detection pair
    for i, track in enumerate(tracks):
        # Predict next position
        pred_x, pred_y, pred_width, pred_height = track.predict()
        pred_center_x = pred_x + pred_width / 2
        pred_center_y = pred_y + pred_height / 2
        
        # Get track velocity and acceleration
        track_vx = track.kf.x[2, 0]
        track_vy = track.kf.x[3, 0]
        track_ax = track.kf.x[4, 0] if track.kf.x.shape[0] > 4 else 0
        track_ay = track.kf.x[5, 0] if track.kf.x.shape[0] > 5 else 0
        track_speed = np.sqrt(track_vx**2 + track_vy**2)
        
        # Get track width and height
        track_width = track.width
        track_height = track.height
        
        # Track's movement direction (unit vector)
        track_direction = np.array([track_vx, track_vy])
        if np.linalg.norm(track_direction) > 0:
            track_direction = track_direction / np.linalg.norm(track_direction)
        
        for j, detection in enumerate(detections):
            # Get detection position
            det_x = detection['x']
            det_y = detection['y']
            det_width = detection['width']
            det_height = detection['height']
            det_center_x = det_x + det_width / 2
            det_center_y = det_y + det_height / 2
            
            # Calculate spatial distance between centers
            dx = pred_center_x - det_center_x
            dy = pred_center_y - det_center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # If distance is too large, skip further calculations
            if distance > max_distance:
                cost_matrix[i, j] = float('inf')
                continue
            
            # Normalize distance
            norm_distance = distance / max_distance
            
            # Calculate direction of detection relative to track
            detection_direction = np.array([det_center_x - pred_center_x, det_center_y - pred_center_y])
            detection_distance = np.linalg.norm(detection_direction)
            
            # Direction similarity (how well the detection aligns with track's velocity)
            direction_similarity = 0.0
            if detection_distance > 0 and track_speed > 0:
                detection_direction = detection_direction / detection_distance
                direction_similarity = np.dot(track_direction, -detection_direction)  # Negative for correct alignment
                direction_similarity = (direction_similarity + 1) / 2  # Scale from [-1, 1] to [0, 1]
            
            # Size similarity (penalize large size changes)
            size_ratio = max(
                track_width/max(1, det_width), 
                det_width/max(1, track_width)
            ) * max(
                track_height/max(1, det_height),
                det_height/max(1, track_height)
            )
            size_similarity = np.exp(-0.5 * (size_ratio - 1))
            
            # Calculate appearance similarity if image is available
            appearance_similarity = 0.0
            if frame_image is not None and track.appearance is not None:
                # Create a temporary track with the detection
                temp_track = BirdTrack(detection, -1)
                temp_track.update_appearance(frame_image)
                
                # Compare appearances using enhanced method
                appearance_similarity = track.compare_appearance(temp_track)
            
            # Calculate costs for different factors
            spatial_cost = norm_distance
            direction_cost = 1.0 - direction_similarity
            size_cost = 1.0 - size_similarity
            appearance_cost = 1.0 - appearance_similarity
            
            # Adaptive weighting based on track characteristics
            # For fast-moving tracks, give more weight to direction
            direction_weight = min(0.3, 0.1 + track_speed / 200.0)
            
            # For inactive tracks, increase appearance weight
            inactive_factor = min(1.0, track.inactive_count / 10.0)
            adj_appearance_weight = appearance_weight * (1.0 + inactive_factor)
            
            # Calculate final weighted cost
            cost = (
                (1.0 - adj_appearance_weight - direction_weight) * spatial_cost + 
                direction_weight * direction_cost + 
                adj_appearance_weight * appearance_cost +
                0.1 * size_cost  # Small weight for size consistency
            )
            
            # Make the cost infinity if the track's predicted motion and the detection distance
            # are completely inconsistent (handles unrealistic accelerations)
            if track_speed > 10 and direction_similarity < 0.2 and distance > track_speed * 2:
                cost = float('inf')
            
            # Assign to cost matrix
            cost_matrix[i, j] = cost
    
    return cost_matrix


def match_detections_to_tracks(tracks, detections, frame_image=None, 
                               max_distance=150, iou_threshold=0.3,
                               appearance_weight=0.3):
    """
    Match detections to existing tracks using Hungarian algorithm.
    
    Args:
        tracks: List of active tracks
        detections: List of detections in current frame
        frame_image: Current frame image for appearance calculation
        max_distance: Maximum distance for matching
        iou_threshold: IoU threshold for matching
        appearance_weight: Weight of appearance features in matching
    
    Returns:
        matched_tracks: List of (track, detection) pairs
        unmatched_tracks: List of tracks with no matching detection
        unmatched_detections: List of detections with no matching track
    """
    if not tracks or not detections:
        return [], tracks, detections
    
    # Calculate cost matrix
    cost_matrix = calculate_cost_matrix(
        tracks, detections, frame_image, max_distance, appearance_weight
    )
    
    # Replace inf values with a very large number for linear_sum_assignment
    large_value = 1e10
    cost_matrix_copy = np.copy(cost_matrix)
    cost_matrix_copy[np.isinf(cost_matrix_copy)] = large_value
    
    # Find minimum cost matching
    from scipy.optimize import linear_sum_assignment
    track_indices, detection_indices = linear_sum_assignment(cost_matrix_copy)
    
    # Collect matches, filtering by cost threshold
    matched_tracks = []
    unmatched_tracks = list(tracks)
    unmatched_detections = list(detections)
    
    for track_idx, det_idx in zip(track_indices, detection_indices):
        # Skip if the original cost was infinite
        if cost_matrix[track_idx, det_idx] >= large_value:
            continue
            
        track = tracks[track_idx]
        detection = detections[det_idx]
        
        # Calculate IoU as additional check
        track_box = (track.x, track.y, track.width, track.height)
        det_box = (detection['x'], detection['y'], detection['width'], detection['height'])
        iou = calculate_iou(track_box, det_box)
        
        # Accept match if IoU is above threshold or distance is small enough
        if iou >= iou_threshold:
            matched_tracks.append((track, detection))
            unmatched_tracks.remove(track)
            unmatched_detections.remove(detection)
    
    return matched_tracks, unmatched_tracks, unmatched_detections


def track_birds(sequence, detections_by_frame, max_distance=150, iou_threshold=0.3, 
               appearance_weight=0.3, inactive_frames=30, min_hits=3):
    """
    Track birds across frames using spatial and appearance matching.
    
    Args:
        sequence: List of (frame_id, image_path) tuples
        detections_by_frame: Dictionary mapping frame IDs to detection lists
        max_distance: Maximum distance for matching tracks to detections
        iou_threshold: IoU threshold for matching
        appearance_weight: Weight of appearance features in matching
        inactive_frames: Maximum number of frames a track can be inactive
        min_hits: Minimum number of detections for a confirmed track
    
    Returns:
        Dictionary mapping frame IDs to track dictionaries
    """
    # Initialize variables
    next_track_id = 0
    active_tracks = []
    all_tracks = []
    tracks_by_frame = {}
    
    # Process each frame
    for frame_id, image_path in tqdm(sequence, desc="Tracking birds"):
        # Initialize tracks for this frame
        tracks_by_frame[frame_id] = []
        
        # Load frame image for appearance features
        frame_image = cv2.imread(image_path)
        
        # Get detections for this frame
        detections = detections_by_frame.get(frame_id, [])
        
        # Predict new locations of existing tracks
        for track in active_tracks:
            track.predict()
        
        # Match detections to existing tracks
        matched_tracks, unmatched_tracks, unmatched_detections = match_detections_to_tracks(
            active_tracks, detections, frame_image, 
            max_distance, iou_threshold, appearance_weight
        )
        
        # Update matched tracks
        for track, detection in matched_tracks:
            track.update(detection, frame_image)
        
        # Mark unmatched tracks as inactive
        for track in unmatched_tracks:
            track.mark_inactive()
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = BirdTrack(detection, next_track_id)
            next_track_id += 1
            
            # Update appearance features
            if frame_image is not None:
                new_track.update_appearance(frame_image)
                
            active_tracks.append(new_track)
            all_tracks.append(new_track)
        
        # Filter active tracks
        active_tracks = [track for track in active_tracks if track.active]
        
        # Add tracks to this frame
        for track in active_tracks:
            if track.confirmed or (track.hit_count >= min_hits):
                # Create track dictionary for this frame
                last_detection = track.get_last_detection()
                track_dict = {
                    'frame': frame_id,
                    'track_id': track.id,
                    'x': track.x,
                    'y': track.y,
                    'width': track.width,
                    'height': track.height,
                    'score': last_detection.get('score', 1.0),
                    'class_id': last_detection.get('class_id', 'bird')
                }
                tracks_by_frame[frame_id].append(track_dict)
    
    return tracks_by_frame, all_tracks


def analyze_tracks(all_tracks):
    """
    Analyze tracking results.
    
    Args:
        all_tracks: List of all tracks
    
    Returns:
        Dictionary with track statistics
    """
    # Filter confirmed tracks
    confirmed_tracks = [track for track in all_tracks if track.confirmed]
    
    if not confirmed_tracks:
        return {
            'total_tracks': 0,
            'avg_detections': 0,
            'avg_duration': 0,
            'max_detections': 0,
            'max_duration': 0
        }
    
    # Calculate track statistics
    track_detections = [len(track.detections) for track in confirmed_tracks]
    track_durations = [max(track.frames) - min(track.frames) + 1 for track in confirmed_tracks]
    
    stats = {
        'total_tracks': len(confirmed_tracks),
        'avg_detections': np.mean(track_detections),
        'avg_duration': np.mean(track_durations),
        'max_detections': np.max(track_detections),
        'max_duration': np.max(track_durations)
    }
    
    if confirmed_tracks:
        # Find longest tracks
        max_det_idx = np.argmax(track_detections)
        max_dur_idx = np.argmax(track_durations)
        
        stats['longest_track_id'] = confirmed_tracks[max_det_idx].id
        stats['longest_duration_id'] = confirmed_tracks[max_dur_idx].id
    
    return stats


def generate_distinct_colors(n):
    """
    Generate n visually distinct colors for tracking visualization.
    Returns a list of (r, g, b) tuples with values from 0-255.
    """
    colors = []
    for i in range(n):
        # Use HSV color space for even distribution
        h = i / n
        s = 0.8 + random.uniform(-0.1, 0.1)  # Add slight randomness
        v = 0.9 + random.uniform(-0.1, 0.1)
        
        # Convert to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    # Shuffle colors to avoid similar colors being adjacent
    random.shuffle(colors)
    return colors


def visualize_tracks(sequence, tracks_by_frame, all_tracks, output_dir, history_frames=20):
    """
    Visualize tracked birds with colored bounding boxes and track IDs.
    
    Args:
        sequence: List of (frame_id, image_path) tuples
        tracks_by_frame: Dictionary mapping frame IDs to track dictionaries
        all_tracks: List of all tracks (for history)
        output_dir: Directory to save visualization results
        history_frames: Number of previous frames to show in trajectory
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, "tracked_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create track ID to color mapping
    track_ids = set()
    for tracks in tracks_by_frame.values():
        for track in tracks:
            track_ids.add(track['track_id'])
    
    track_colors = generate_distinct_colors(len(track_ids))
    color_map = {track_id: track_colors[i] for i, track_id in enumerate(sorted(track_ids))}
    
    # Build track history for each track
    track_history = {track_id: [] for track_id in track_ids}
    
    # Process each frame
    for frame_id, image_path in tqdm(sequence, desc="Visualizing tracks"):
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Update track history
        if frame_id in tracks_by_frame:
            for track in tracks_by_frame[frame_id]:
                track_id = track['track_id']
                x = track['x'] + track['width'] // 2  # Center x
                y = track['y'] + track['height'] // 2  # Center y
                track_history[track_id].append((frame_id, x, y))
        
        # Draw track history (trajectories)
        for track_id, history in track_history.items():
            # Filter history to only include frames up to current frame
            # and limit to the specified history length
            history = [(f, x, y) for f, x, y in history if f < frame_id]
            if not history:
                continue
                
            # Only show the most recent frames
            history = history[-history_frames:]
            
            # Draw trajectory
            points = np.array([(x, y) for _, x, y in history], np.int32)
            if len(points) >= 2:
                cv2.polylines(
                    image_rgb, [points], False, color_map[track_id], 2
                )
        
        # Draw current tracks
        if frame_id in tracks_by_frame:
            for track in tracks_by_frame[frame_id]:
                track_id = track['track_id']
                x = int(track['x'])
                y = int(track['y'])
                w = int(track['width'])
                h = int(track['height'])
                color = color_map[track_id]
                
                # Draw bounding box
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 2)
                
                # Draw track ID with background for readability
                id_text = f"ID: {track_id}"
                
                # Add background rectangle for text
                (text_width, text_height), _ = cv2.getTextSize(
                    id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    image_rgb,
                    (x, y - text_height - 10),
                    (x + text_width + 10, y),
                    color,
                    -1  # Filled rectangle
                )
                
                # Draw text in white
                cv2.putText(
                    image_rgb, id_text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                # Draw center point
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(image_rgb, (center_x, center_y), 4, color, -1)
                
        # Save visualization
        output_path = os.path.join(vis_dir, f"{frame_id:02d}_tracked.jpg")
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    print(f"Track visualizations saved to {vis_dir}")


def save_tracks_to_csv(tracks_by_frame, output_path):
    """
    Save tracking results to a CSV file.
    
    Args:
        tracks_by_frame: Dictionary mapping frame IDs to track dictionaries
        output_path: Path to save CSV file
    """
    # Flatten track dictionaries
    all_tracks = []
    for frame_id, tracks in tracks_by_frame.items():
        all_tracks.extend(tracks)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_tracks)
    
    # Reorder columns
    cols = ['frame', 'track_id', 'x', 'y', 'width', 'height', 'score', 'class_id']
    df = df[cols]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} track entries to {output_path}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    sequence = load_images(args.images_dir)
    print(f"Found {len(sequence)} images in sequence")
    
    # Load detections
    print(f"Loading detections from {args.detections_csv}")
    detections_by_frame = load_detections(args.detections_csv)
    
    # Get number of frames with detections
    frames_with_detections = len(detections_by_frame)
    print(f"Loaded detections for {frames_with_detections} frames")
    
    # Track birds across frames
    print("Tracking birds across frames...")
    tracks_by_frame, all_tracks = track_birds(
        sequence,
        detections_by_frame,
        max_distance=args.max_distance,
        iou_threshold=args.iou_threshold,
        appearance_weight=args.appearance_weight,
        inactive_frames=args.inactive_frames,
        min_hits=args.min_hits
    )
    
    # Analyze tracking results
    stats = analyze_tracks(all_tracks)
    print("Track analysis:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Average track length: {stats['avg_detections']:.2f} detections")
    print(f"  Average track duration: {stats['avg_duration']:.2f} frames")
    if stats['total_tracks'] > 0:
        print(f"  Longest track: {stats['max_detections']} detections (ID: {stats['longest_track_id']})")
        print(f"  Longest duration: {stats['max_duration']} frames (ID: {stats['longest_duration_id']})")
    
    # Save results to CSV
    csv_path = os.path.join(args.output_dir, "tracked_detections.csv")
    save_tracks_to_csv(tracks_by_frame, csv_path)
    
    # Visualize tracks
    print("Visualizing tracks...")
    visualize_tracks(
        sequence, 
        tracks_by_frame, 
        all_tracks, 
        args.output_dir,
        history_frames=args.history_frames
    )
    
    print("Bird tracking completed!")


if __name__ == "__main__":
    main() 