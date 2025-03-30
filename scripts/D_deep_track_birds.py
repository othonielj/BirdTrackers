#!/usr/bin/env python
"""
Deep Learning Enhanced Bird Tracker

This script implements a bird tracking system using deep learning features:
- Uses CoreML-optimized MobileNetV2 for bird appearance embeddings
- Combines deep features with spatial and color-based tracking
- Enhanced Kalman filtering for motion prediction
- Optimized for Apple Silicon with efficient feature extraction

Author: DETracker Team
Date: 2023
"""

import os
import glob
import argparse
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import colorsys
import random
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import tensorflow as tf
import coremltools as ct
from typing import Dict, List, Tuple, Optional, Union

# Global variables for caching
model = None
feature_cache = {}
frame_cache = {}

# Constants
EMBEDDING_DIM = 1280  # Output dimension of MobileNetV2 features
FEATURE_EXTRACT_FREQUENCY = 3  # Extract features every N frames to save computation

def parse_args():
    """
    Parse command line arguments for the deep learning enhanced bird tracker.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Track birds using deep learning features and Kalman filtering")
    
    parser.add_argument("--detections_csv", type=str, required=True,
                        help="Path to CSV file with bird detections")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output results")
    
    # Tracking parameters
    parser.add_argument("--max_distance", type=float, default=150,
                        help="Maximum distance for matching detections to tracks")
    parser.add_argument("--iou_threshold", type=float, default=0.3,
                        help="Minimum IoU for detection matching")
    parser.add_argument("--appearance_weight", type=float, default=0.3,
                        help="Weight for appearance features in matching cost")
    parser.add_argument("--deep_feature_weight", type=float, default=0.4,
                        help="Weight for deep learning features in matching cost")
    parser.add_argument("--inactive_frames", type=int, default=30,
                        help="Maximum number of frames a track can be inactive")
    parser.add_argument("--min_hits", type=int, default=3,
                        help="Minimum number of hits for a track to be considered valid")
    parser.add_argument("--history_frames", type=int, default=30,
                        help="Number of frames to show track trajectory in visualization")
    
    # Feature extraction parameters
    parser.add_argument("--use_deep_features", type=bool, default=True,
                        help="Whether to use deep learning features")
    parser.add_argument("--feature_cache_size", type=int, default=50,
                        help="Number of frames to keep in feature cache")
    
    return parser.parse_args()


def load_images(input_dir):
    """
    Load image paths from the input directory.
    
    Args:
        input_dir: Directory containing images
        
    Returns:
        List of image paths sorted by frame number
    """
    # Get all image files in the directory
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
    
    # Sort images to ensure sequential order
    image_paths.sort()
    
    return image_paths


def load_detections(csv_path):
    """
    Load bird detections from CSV file.
    
    Args:
        csv_path: Path to the CSV file with detections
        
    Returns:
        Dictionary mapping frame_id to list of detections
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize dictionary to store detections by frame
    detections_by_frame = {}
    
    # Group by frame
    for frame_id, group in df.groupby('frame'):
        detections = []
        
        # Convert each row to a detection dictionary
        for _, row in group.iterrows():
            detection = {
                'x': float(row['x']),
                'y': float(row['y']),
                'width': float(row['width']),
                'height': float(row['height']),
                'score': float(row['score']) if 'score' in row else 0.9,
                'class_id': row['class_id'] if 'class_id' in row else 'bird'
            }
            detections.append(detection)
        
        # Store in dictionary
        detections_by_frame[int(frame_id)] = detections
    
    return detections_by_frame


def load_feature_extractor():
    """
    Load and prepare the feature extractor model.
    Uses a pre-trained MobileNetV2 model optimized for Apple Silicon if possible.
    
    Returns:
        Feature extractor model
    """
    global model
    
    if model is not None:
        return model
    
    print("Loading feature extraction model...")
    
    try:
        # First try to load as CoreML model for Apple Silicon optimization
        use_coreml = False
        
        # Check if we're on macOS and can use CoreML
        if 'darwin' in os.uname().sysname.lower():
            use_coreml = True
            print("Running on macOS, attempting to use CoreML optimization")
        
        if use_coreml:
            try:
                # Load base MobileNetV2 model
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(128, 128, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
                
                # Create a feature extraction model
                feature_model = tf.keras.Model(
                    inputs=base_model.input,
                    outputs=base_model.output
                )
                
                # Convert to CoreML model if on macOS
                print("Converting to CoreML model for Apple Silicon optimization...")
                mlmodel = ct.convert(
                    feature_model,
                    source='tensorflow',
                    inputs=[ct.TensorType(shape=(1, 128, 128, 3))]
                )
                
                # Save temporarily and load
                temp_model_path = "temp_bird_feature_model.mlmodel"
                mlmodel.save(temp_model_path)
                model = ct.models.MLModel(temp_model_path)
                
                # Delete temporary file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                
                print("Successfully loaded CoreML-optimized model")
                return model
                
            except Exception as e:
                print(f"CoreML conversion failed: {e}")
                print("Falling back to standard TensorFlow model")
        
        # Fallback to standard TensorFlow model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(128, 128, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        model = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.output
        )
        
        print("Successfully loaded TensorFlow model")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def extract_deep_features(image, detection, frame_id, feature_cache_size=50):
    """
    Extract deep learning features from a detection in an image.
    Uses caching to avoid redundant computation.
    
    Args:
        image: Full frame image
        detection: Dictionary with detection coordinates
        frame_id: Current frame ID for caching
        feature_cache_size: Size of feature cache (in frames)
        
    Returns:
        Feature vector (numpy array)
    """
    global feature_cache, frame_cache
    
    # Generate a unique key for this detection
    det_key = f"{frame_id}_{int(detection['x'])}_{int(detection['y'])}_{int(detection['width'])}_{int(detection['height'])}"
    
    # Check if we have a cached feature
    if det_key in feature_cache:
        return feature_cache[det_key]
    
    # Cache management (keep last N frames)
    current_frames = set([k.split('_')[0] for k in feature_cache.keys()])
    if len(current_frames) > feature_cache_size:
        # Remove oldest frames
        oldest_frame = min(map(int, current_frames))
        keys_to_remove = [k for k in feature_cache.keys() if k.startswith(f"{oldest_frame}_")]
        for k in keys_to_remove:
            del feature_cache[k]
    
    # Extract the bird region
    x, y, w, h = int(detection['x']), int(detection['y']), int(detection['width']), int(detection['height'])
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    
    # Extract the patch
    bird_patch = image[y:y+h, x:x+w]
    
    if bird_patch.size == 0:
        # Return zeros if patch is empty
        features = np.zeros(EMBEDDING_DIM)
        feature_cache[det_key] = features
        return features
    
    try:
        # Load the model if needed
        if model is None:
            load_feature_extractor()
        
        # Resize to model input size (128x128)
        resized = cv2.resize(bird_patch, (128, 128))
        
        # Convert BGR to RGB (TensorFlow models expect RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Preprocess for the model
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)
        
        # Extract features
        if hasattr(model, 'predict'):
            # TensorFlow model
            features = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)[0]
        else:
            # CoreML model
            results = model.predict({'input_1': np.expand_dims(preprocessed, axis=0)})
            features = results['Identity']
            
        # Normalize the features
        features = features / np.linalg.norm(features)
        
        # Cache the result
        feature_cache[det_key] = features
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        features = np.zeros(EMBEDDING_DIM)
        feature_cache[det_key] = features
        return features

class BirdTrack:
    """
    Class to represent a bird track with Kalman filtering and deep appearance features
    """
    
    def __init__(self, track_id, detection, frame_id, frame_image=None):
        """
        Initialize a new track with a detection
        
        Args:
            track_id: Unique identifier for this track
            detection: Initial detection dictionary
            frame_id: Frame ID for initial detection
            frame_image: Optional image for appearance initialization
        """
        self.track_id = track_id
        self.detections = [detection]
        self.frames = [frame_id]
        
        # Track state
        self.hit_count = 1  # Number of successful matches (initialize with 1 for first detection)
        self.inactive_count = 0  # Number of frames since last match
        self.is_active = True
        
        # Initialize Kalman filter
        self.kf = self.initialize_kalman_filter()
        
        # Initialize state with first detection
        x = detection['x'] + detection['width']/2  # center x
        y = detection['y'] + detection['height']/2  # center y
        w = detection['width']
        h = detection['height']
        vx = 0  # Initial velocity
        vy = 0
        
        # Set initial state [x, y, width, height, vx, vy, ax, ay]
        self.kf.x = np.array([x, y, w, h, vx, vy, 0, 0])
        
        # Store velocity for convenience
        self.vx = vx
        self.vy = vy
        
        # Appearance features
        self.appearance_history = []
        self.deep_features = []
        self.avg_deep_feature = None
        
        # Initialize appearance if image is provided
        if frame_image is not None:
            self.update_appearance(frame_image)
            self.update_deep_features(frame_image, frame_id)
    
    def initialize_kalman_filter(self):
        """
        Initialize a Kalman filter for tracking the bird.
        Uses a constant acceleration model with 8 state variables:
        [x, y, width, height, vx, vy, ax, ay]
        
        Returns:
            Configured KalmanFilter object
        """
        # Create Kalman filter with 8 state variables and 4 measurement variables
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant acceleration model)
        kf.F = np.eye(8)
        dt = 1.0  # Time step
        
        # Position update with velocity and acceleration
        kf.F[0, 4] = dt     # x += vx * dt
        kf.F[1, 5] = dt     # y += vy * dt
        kf.F[4, 6] = dt     # vx += ax * dt
        kf.F[5, 7] = dt     # vy += ay * dt
        
        # Measurement function (we only measure position and size)
        kf.H = np.zeros((4, 8))
        kf.H[0, 0] = 1.0    # x
        kf.H[1, 1] = 1.0    # y
        kf.H[2, 2] = 1.0    # width
        kf.H[3, 3] = 1.0    # height
        
        # Measurement noise
        kf.R = np.eye(4) * 10.0
        
        # Process noise
        kf.Q = np.eye(8)
        # Position and size have less noise
        kf.Q[0:4, 0:4] *= 1.0
        # Velocity has moderate noise
        kf.Q[4:6, 4:6] *= 10.0
        # Acceleration has more noise
        kf.Q[6:8, 6:8] *= 100.0
        
        # Initial uncertainty
        kf.P = np.eye(8) * 100.0
        
        return kf
    
    def predict(self):
        """
        Predict the next position of the bird using the Kalman filter.
        
        Returns:
            Tuple of (x, y, width, height) for predicted position
        """
        # Predict next state
        self.kf.predict()
        
        # Get center position from state
        center_x = self.kf.x[0]
        center_y = self.kf.x[1]
        
        # Update velocity from Kalman filter
        self.vx = self.kf.x[4]
        self.vy = self.kf.x[5]
        
        # Apply physical constraints to predicted flight path
        self.apply_flight_constraints()
        
        # Return predicted bounding box (x, y, width, height)
        return (
            self.kf.x[0] - self.kf.x[2]/2,  # top-left x
            self.kf.x[1] - self.kf.x[3]/2,  # top-left y
            self.kf.x[2],                   # width
            self.kf.x[3]                    # height
        )
    
    def apply_flight_constraints(self):
        """
        Apply physical constraints to the predicted flight path.
        Birds have maximum speeds and tend to maintain momentum.
        """
        # Get velocity from Kalman filter
        vx = self.kf.x[4]
        vy = self.kf.x[5]
        
        # Calculate speed
        speed = np.sqrt(vx**2 + vy**2)
        
        # Maximum speed constraint (in pixels per frame)
        max_speed = 30.0
        
        # If speed exceeds maximum, scale it down
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale
            
            # Update Kalman state with constrained velocity
            self.kf.x[4] = vx
            self.kf.x[5] = vy
            
            # Update stored velocity
            self.vx = vx
            self.vy = vy
    
    def update(self, detection, frame_id, frame_image=None):
        """
        Update track with a new detection.
        
        Args:
            detection: Dictionary with detection information
            frame_id: Frame ID for this detection
            frame_image: Optional image for appearance update
        """
        # Add detection and frame to history
        self.detections.append(detection)
        self.frames.append(frame_id)
        
        # Update activity states
        self.hit_count += 1
        self.inactive_count = 0
        
        # Get position from current detection
        box = np.array([
            detection['x'] + detection['width']/2,  # center x
            detection['y'] + detection['height']/2, # center y
            detection['width'],
            detection['height']
        ])
        
        # Update kalman filter with new measurement
        self.kf.update(box)
        
        # Get post-update state (improved estimation)
        state = self.kf.x
        
        # Update stored velocity
        self.vx = state[4]
        self.vy = state[5]
        
        # Update detection with Kalman-corrected values
        detection['x'] = state[0] - state[2]/2  # center x -> top left x
        detection['y'] = state[1] - state[3]/2  # center y -> top left y
        detection['width'] = state[2]
        detection['height'] = state[3]
        detection['vx'] = state[4]
        detection['vy'] = state[5]
        
        # Update appearance features if image is provided
        if frame_image is not None:
            self.update_appearance(frame_image)
            
            # Only update deep features periodically to save computation
            if self.hit_count % FEATURE_EXTRACT_FREQUENCY == 0:
                self.update_deep_features(frame_image, frame_id)
    
    def update_appearance(self, image):
        """
        Update appearance features from the current detection.
        
        Args:
            image: Full frame image
        """
        if not self.detections:
            return
            
        # Get the most recent detection
        last_det = self.detections[-1]
        
        # Extract the bird region
        x, y = int(last_det['x']), int(last_det['y'])
        w, h = int(last_det['width']), int(last_det['height'])
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Extract the bird patch
        if w <= 0 or h <= 0:
            return
            
        bird_patch = image[y:y+h, x:x+w]
        if bird_patch.size == 0:
            return
            
        # Extract enhanced features
        features = self.extract_enhanced_features(bird_patch)
        
        # Add to appearance history
        self.appearance_history.append(features)
        
        # Limit history size
        if len(self.appearance_history) > 5:
            self.appearance_history.pop(0)
    
    def extract_enhanced_features(self, patch):
        """
        Extract enhanced appearance features from an image patch.
        Combines histogram features with grid-based intensity features.
        
        Args:
            patch: Image patch containing the bird
            
        Returns:
            Feature vector
        """
        # Resize for consistent features
        resized = cv2.resize(patch, (32, 32))
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Extract histogram features (3 channels x 32 bins = 96 dimensions)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Reshape and combine histograms
        h_hist = h_hist.reshape(-1)
        s_hist = s_hist.reshape(-1)
        v_hist = v_hist.reshape(-1)
        
        # Extract grid-based average intensity features (4x4 grid = 16 dimensions)
        grid_size = 4
        cell_h, cell_w = resized.shape[0] // grid_size, resized.shape[1] // grid_size
        grid_features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = resized[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                avg_intensity = np.mean(cell, axis=(0, 1)) / 255.0  # Normalize to [0,1]
                grid_features.extend(avg_intensity)
                
        # Combine all features
        combined_features = np.concatenate([h_hist, s_hist, v_hist, grid_features])
        
        return combined_features
    
    def compare_appearance(self, other_features):
        """
        Compare appearance features with other features.
        Uses a combination of histogram correlation and Euclidean distance.
        
        Args:
            other_features: Feature vector to compare against
            
        Returns:
            Similarity score (0-1 range, higher is more similar)
        """
        if len(self.appearance_history) == 0:
            return 0.0
            
        # Get most recent appearance features
        features = self.appearance_history[-1]
        
        # Split features into histogram and grid components
        hist_features = features[:96]  # First 96 dimensions are histograms
        grid_features = features[96:]  # Remaining are grid features
        
        other_hist = other_features[:96]
        other_grid = other_features[96:]
        
        # Compare histograms using correlation
        hist_corr = 0
        try:
            # Split histograms into H, S, V components for better comparison
            h_hist1 = hist_features[:32].reshape(-1, 1).astype(np.float32)
            s_hist1 = hist_features[32:64].reshape(-1, 1).astype(np.float32)
            v_hist1 = hist_features[64:96].reshape(-1, 1).astype(np.float32)
            
            h_hist2 = other_hist[:32].reshape(-1, 1).astype(np.float32)
            s_hist2 = other_hist[32:64].reshape(-1, 1).astype(np.float32)
            v_hist2 = other_hist[64:96].reshape(-1, 1).astype(np.float32)
            
            # Calculate correlation for each channel
            h_corr = cv2.compareHist(h_hist1, h_hist2, cv2.HISTCMP_CORREL)
            s_corr = cv2.compareHist(s_hist1, s_hist2, cv2.HISTCMP_CORREL)
            v_corr = cv2.compareHist(v_hist1, v_hist2, cv2.HISTCMP_CORREL)
            
            # Weight hue more heavily (more distinctive for birds)
            hist_corr = 0.5 * h_corr + 0.25 * s_corr + 0.25 * v_corr
            
            # Ensure in range [0, 1]
            hist_corr = max(0.0, min(1.0, hist_corr))
            
        except Exception as e:
            print(f"Error comparing histograms: {e}")
            hist_corr = 0.0
        
        # Compare grid features using Euclidean distance
        grid_dist = 0
        try:
            grid_dist = np.linalg.norm(grid_features - other_grid)
            max_dist = np.sqrt(len(grid_features) * 4)  # Maximum possible distance (assuming values in [0,1])
            grid_sim = 1.0 - min(1.0, grid_dist / max_dist)
        except Exception as e:
            print(f"Error comparing grid features: {e}")
            grid_sim = 0.0
        
        # Combine similarities with weights
        combined_sim = 0.7 * hist_corr + 0.3 * grid_sim
        
        return combined_sim
    
    def update_deep_features(self, image, frame_id, feature_cache_size=50):
        """
        Updates deep features using the provided image patch for the most recent detection.
        
        Args:
            image: Full frame image
            frame_id: Current frame ID for caching
            feature_cache_size: Size of feature cache (in frames)
        """
        if len(self.detections) == 0:
            return
            
        last_detection = self.detections[-1]
        
        # Extract features
        features = extract_deep_features(image, last_detection, frame_id, feature_cache_size)
        
        # Update feature history
        self.deep_features.append(features)
        
        # Compute average feature if we have more than one
        if len(self.deep_features) > 1:
            self.avg_deep_feature = np.mean(self.deep_features, axis=0)
            # Normalize
            norm = np.linalg.norm(self.avg_deep_feature)
            if norm > 0:
                self.avg_deep_feature = self.avg_deep_feature / norm
        else:
            self.avg_deep_feature = features
    
    def mark_inactive(self):
        """Mark track as inactive but continue prediction."""
        self.inactive_count += 1
        
        # Still perform Kalman prediction even when inactive
        self.predict()
        
        if self.inactive_count > 30:  # Maximum inactive frames
            self.is_active = False
    
    def get_last_detection(self):
        """Get the most recent detection."""
        return self.detections[-1]
    
    def get_last_position(self):
        """Get the most recent position."""
        if self.detections:
            last_det = self.detections[-1]
            return (
                last_det['x'],
                last_det['y'],
                last_det['width'],
                last_det['height']
            )
        else:
            return (
                self.kf.x[0] - self.kf.x[2]/2,  # top-left x
                self.kf.x[1] - self.kf.x[3]/2,  # top-left y
                self.kf.x[2],                   # width
                self.kf.x[3]                    # height
            )
    
    def get_predicted_position(self):
        """Get the predicted next position."""
        center_x = self.kf.x[0]
        center_y = self.kf.x[1]
        return center_x - self.kf.x[4]/2, center_y - self.kf.x[5]/2, self.kf.x[4], self.kf.x[5]
    
    def get_center(self):
        """Get the center point of the bounding box."""
        return self.kf.x[0], self.kf.x[1]

    def compare_deep_features(self, features):
        """
        Compare deep features with provided feature vector.
        Uses cosine similarity for comparison.
        
        Args:
            features: Feature vector to compare against
            
        Returns:
            Similarity score (0-1 range, higher is more similar)
        """
        if not self.deep_features or self.avg_deep_feature is None:
            return 0.0
            
        # Calculate cosine similarity with average deep feature
        norm1 = np.linalg.norm(self.avg_deep_feature)
        norm2 = np.linalg.norm(features)
        
        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(self.avg_deep_feature, features) / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))  # Ensure range [0,1]
        
        return 0.0

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


def calculate_cost_matrix(tracks, detections, frame_image=None, frame_id=None, 
                         max_distance=150, appearance_weight=0.3, deep_feature_weight=0.4):
    """
    Calculate cost matrix for matching tracks to detections.
    
    Args:
        tracks: List of active tracks
        detections: List of detections in current frame
        frame_image: Optional frame image for appearance features
        frame_id: Current frame ID
        max_distance: Maximum distance for matching
        appearance_weight: Weight for appearance in cost calculation
        deep_feature_weight: Weight for deep features in cost calculation
        
    Returns:
        Cost matrix (tracks x detections)
    """
    num_tracks = len(tracks)
    num_detections = len(detections)
    
    # Initialize cost matrix with maximum distance
    cost_matrix = np.ones((num_tracks, num_detections)) * max_distance
    
    # Cache for detection features to avoid redundant computation
    detection_features = {}
    
    # Calculate costs for each track-detection pair
    for i, track in enumerate(tracks):
        # Get predicted position from track
        predicted_pos = track.predict()
        track_x = predicted_pos[0] + predicted_pos[2]/2  # center x
        track_y = predicted_pos[1] + predicted_pos[3]/2  # center y
        
        # Track size for bbox scaling
        track_width = predicted_pos[2]
        track_height = predicted_pos[3]
        
        # Get track velocity
        track_vx = track.vx
        track_vy = track.vy
        track_speed = np.sqrt(track_vx**2 + track_vy**2)
        
        # Normalize direction vector if speed is significant
        if track_speed > 1.0:
            direction_x = track_vx / track_speed
            direction_y = track_vy / track_speed
        else:
            direction_x = 0
            direction_y = 0
        
        # Process each detection
        for j, detection in enumerate(detections):
            # Get detection center
            det_x = detection['x'] + detection['width']/2
            det_y = detection['y'] + detection['height']/2
            
            # Calculate spatial distance
            dx = det_x - track_x
            dy = det_y - track_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Penalize detections that are in the opposite direction of movement
            if track_speed > 5.0:  # Only consider direction for fast-moving birds
                # Calculate how aligned the detection is with the movement direction
                movement_dot = dx*direction_x + dy*direction_y
                
                # If moving away from the predicted path, increase cost
                if movement_dot < 0:
                    direction_penalty = 1.5  # Penalty factor for opposite direction
                    distance *= direction_penalty
            
            # Calculate appearance similarity if image is provided
            appearance_similarity = 0.0
            if frame_image is not None and track.appearance_history:
                # Extract features from current detection
                x, y, w, h = int(detection['x']), int(detection['y']), int(detection['width']), int(detection['height'])
                
                # Ensure coordinates are within image bounds
                height, width = frame_image.shape[:2]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))
                
                # Extract patch and calculate features
                if w > 0 and h > 0:
                    bird_patch = frame_image[y:y+h, x:x+w]
                    if bird_patch.size > 0:
                        # Extract histogram and normalized pixels
                        hist_features = track.extract_enhanced_features(bird_patch)
                        
                        # Compare with track's appearance
                        similarities = []
                        for track_hist in track.appearance_history:
                            similarity = track.compare_appearance(hist_features)
                            similarities.append(similarity)
                        
                        # Use maximum similarity
                        appearance_similarity = max(similarities) if similarities else 0.0
            
            # Calculate deep feature similarity if available
            deep_similarity = 0.0
            if frame_image is not None and frame_id is not None and track.deep_features and track.avg_deep_feature is not None:
                # Extract deep features if not already cached
                if j not in detection_features:
                    features = extract_deep_features(frame_image, detection, frame_id)
                    detection_features[j] = features
                else:
                    features = detection_features[j]
                
                # Calculate cosine similarity with track's average deep feature
                feat_norm = np.linalg.norm(features)
                track_norm = np.linalg.norm(track.avg_deep_feature)
                
                if feat_norm > 0 and track_norm > 0:
                    deep_similarity = np.dot(features, track.avg_deep_feature) / (feat_norm * track_norm)
                    deep_similarity = max(0.0, deep_similarity)  # Ensure non-negative
            
            # Calculate final cost as weighted sum of distance and feature similarities
            # Lower cost = better match
            # Convert similarities to costs (1 - similarity)
            appearance_cost = 1.0 - appearance_similarity
            deep_cost = 1.0 - deep_similarity
            
            # Scale distance to [0, 1] range relative to max_distance
            normalized_distance = min(1.0, distance / max_distance)
            
            # Calculate weighted cost
            # More weight to spatial distance for high-speed tracks
            spatial_weight = min(0.7, 0.3 + track_speed / 100.0)
            
            # Adjust weights to ensure they sum to 1
            remaining_weight = 1.0 - spatial_weight
            adj_appearance_weight = remaining_weight * (appearance_weight / (appearance_weight + deep_feature_weight))
            adj_deep_weight = remaining_weight * (deep_feature_weight / (appearance_weight + deep_feature_weight))
            
            # Final cost calculation
            cost = (
                spatial_weight * normalized_distance +
                adj_appearance_weight * appearance_cost +
                adj_deep_weight * deep_cost
            )
            
            # Scale back to original distance range
            cost *= max_distance
            
            # Update cost matrix
            cost_matrix[i, j] = cost
    
    return cost_matrix


def match_detections_to_tracks(tracks, detections, frame_image=None, frame_id=None, 
                              max_distance=150, iou_threshold=0.3, 
                              appearance_weight=0.3, deep_feature_weight=0.4):
    """
    Match detections to existing tracks using the Hungarian algorithm.
    
    Args:
        tracks: List of active tracks
        detections: List of detections in current frame
        frame_image: Optional frame image for appearance features
        frame_id: Current frame ID for feature extraction
        max_distance: Maximum distance for matching
        iou_threshold: Minimum IoU for matching
        appearance_weight: Weight for appearance in cost calculation
        deep_feature_weight: Weight for deep features in cost calculation
        
    Returns:
        Tuple of (matched_tracks, unmatched_tracks, unmatched_detections)
        Each element in matched_tracks is a tuple of (track_idx, detection_idx)
    """
    if not tracks or not detections:
        # If no tracks or detections, all are unmatched
        return [], list(range(len(tracks))), list(range(len(detections)))
        
    # Calculate cost matrix
    cost_matrix = calculate_cost_matrix(
        tracks, detections, frame_image, frame_id,
        max_distance, appearance_weight, deep_feature_weight
    )
    
    # Copy cost matrix for Hungarian algorithm (which modifies the matrix)
    cost_matrix_copy = cost_matrix.copy()
    
    # Find minimum cost matching
    track_indices, detection_indices = linear_sum_assignment(cost_matrix_copy)
    
    # Create lists for matching results
    matched_tracks = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))
    
    # Process matches, checking cost thresholds
    for track_idx, detection_idx in zip(track_indices, detection_indices):
        # Check if cost is too high (no valid match)
        if cost_matrix[track_idx, detection_idx] > max_distance:
            continue
            
        # Get track and detection
        track = tracks[track_idx]
        detection = detections[detection_idx]
        
        # Calculate IoU as additional check
        x1, y1, w1, h1 = track.get_last_position()
        x2, y2, w2, h2 = detection['x'], detection['y'], detection['width'], detection['height']
        
        box1 = (x1, y1, w1, h1)
        box2 = (x2, y2, w2, h2)
        iou = calculate_iou(box1, box2)
        
        # If IoU check passes, consider it a match
        if iou >= iou_threshold or cost_matrix[track_idx, detection_idx] < max_distance * 0.7:
            matched_tracks.append((track_idx, detection_idx))
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(detection_idx)
    
    return matched_tracks, unmatched_tracks, unmatched_detections


def track_birds(sequence, detections_by_frame, max_distance=150, iou_threshold=0.3, appearance_weight=0.3, deep_feature_weight=0.4, inactive_frames=30, min_hits=3, use_deep_features=True, feature_cache_size=50):
    """
    Track birds across frames using deep features and Kalman filtering.
    
    Args:
        sequence: List of image paths
        detections_by_frame: Dictionary mapping frame_id to list of detections
        max_distance: Maximum distance for matching
        iou_threshold: Minimum IoU for matching
        appearance_weight: Weight for appearance in cost matrix
        deep_feature_weight: Weight for deep features in cost matrix
        inactive_frames: Maximum number of frames a track can be inactive
        min_hits: Minimum number of hits for a track to be considered valid
        use_deep_features: Whether to use deep features for matching
        feature_cache_size: Size of feature cache (in frames)
        
    Returns:
        Dictionary mapping frame_id to list of track dictionaries
        List of all tracks
    """
    frame_count = len(sequence)
    active_tracks = []
    all_tracks = []
    next_track_id = 0
    tracks_by_frame = {}
    
    # Process each frame
    for frame_id in tqdm(range(frame_count), desc="Tracking birds"):
        image_path = sequence[frame_id]
        
        # Skip if no detections for this frame
        if frame_id not in detections_by_frame:
            continue
            
        # Get current frame's detections
        detections = detections_by_frame[frame_id]
        
        # Load image if needed for appearance features
        frame_image = None
        if use_deep_features:
            frame_image = cv2.imread(image_path)
            if frame_image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
        
        # Predict new locations of existing tracks
        for track in active_tracks:
            track.predict()
        
        # Match detections to existing tracks
        matched_tracks, unmatched_tracks, unmatched_detections = match_detections_to_tracks(
            active_tracks, detections, frame_image, frame_id,
            max_distance, iou_threshold, appearance_weight, deep_feature_weight
        )
        
        # Update matched tracks
        for track_idx, detection_idx in matched_tracks:
            track = active_tracks[track_idx]
            detection = detections[detection_idx]
            track.update(detection, frame_id, frame_image)
        
        # Mark unmatched tracks as inactive
        for track_idx in unmatched_tracks:
            active_tracks[track_idx].mark_inactive()
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            new_track = BirdTrack(next_track_id, detection, frame_id, frame_image)
            active_tracks.append(new_track)
            all_tracks.append(new_track)
            next_track_id += 1
        
        # Remove tracks that have been inactive for too long
        active_tracks = [track for track in active_tracks if track.inactive_count <= inactive_frames]
        
        # Store track information for this frame
        tracks_by_frame[frame_id] = []
        for track in active_tracks:
            if track.hit_count >= min_hits:
                # Get position from last detection
                if track.detections:
                    last_det = track.detections[-1]
                    x, y = last_det['x'], last_det['y']
                    w, h = last_det['width'], last_det['height']
                else:
                    x, y, w, h = track.get_last_position()
                
                # Create track dictionary
                track_dict = {
                    'track_id': track.track_id,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'frame': frame_id,
                    'score': track.detections[-1]['score'] if track.detections else 0.5,
                    'class_id': 'bird',
                    'active': track.is_active,
                    'inactive_count': track.inactive_count
                }
                tracks_by_frame[frame_id].append(track_dict)
    
    return tracks_by_frame, all_tracks


def analyze_tracks(all_tracks):
    """Analyze track statistics."""
    if not all_tracks:
        return {
            'total_tracks': 0,
            'avg_length': 0,
            'avg_duration': 0,
            'longest_track': 0,
            'longest_duration': 0
        }
    
    # Calculate track statistics
    track_lengths = [len(track.detections) for track in all_tracks if track.hit_count >= 3]
    track_durations = []
    for track in all_tracks:
        if track.hit_count >= 3:
            frames = track.frames
            duration = max(frames) - min(frames) + 1 if frames else 0
            track_durations.append(duration)
    
    # Identify longest tracks
    longest_idx = np.argmax(track_lengths) if track_lengths else -1
    longest_duration_idx = np.argmax(track_durations) if track_durations else -1
    
    # Compute statistics
    stats = {
        'total_tracks': len([t for t in all_tracks if t.hit_count >= 3]),
        'avg_length': np.mean(track_lengths) if track_lengths else 0,
        'avg_duration': np.mean(track_durations) if track_durations else 0,
        'longest_track': max(track_lengths) if track_lengths else 0,
        'longest_track_id': all_tracks[longest_idx].track_id if longest_idx >= 0 else -1,
        'longest_duration': max(track_durations) if track_durations else 0,
        'longest_duration_id': all_tracks[longest_duration_idx].track_id if longest_duration_idx >= 0 else -1
    }
    
    return stats


def save_tracks_to_csv(tracks_by_frame, output_path):
    """
    Save track data to a CSV file.
    
    Args:
        tracks_by_frame: Dictionary mapping frame_id to list of track dictionaries
        output_path: Path to save the CSV file
    
    Returns:
        Number of track entries saved
    """
    # Collect all track data
    all_track_data = []
    
    # Process each frame's tracks
    for frame_id, tracks in tracks_by_frame.items():
        for track in tracks:
            # Create data entry for this track
            track_data = {
                'frame': frame_id,
                'track_id': track['track_id'],
                'x': track['x'],
                'y': track['y'],
                'width': track['width'],
                'height': track['height'],
                'score': track['score'],
                'class_id': track['class_id'],
                'active': 1 if track['active'] else 0,
                'inactive_count': track['inactive_count']
            }
            all_track_data.append(track_data)
    
    # Create DataFrame from track data
    if not all_track_data:
        print("Warning: No track data to save")
        return 0
        
    df = pd.DataFrame(all_track_data)
    
    # Sort by frame and track_id
    df = df.sort_values(['frame', 'track_id'])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return len(df)


def draw_tracks(image, tracks, colormap=None, show_ids=True, show_inactive=True, history_frames=20):
    """
    Draw tracks on an image.
    
    Args:
        image: Image to draw on
        tracks: List of tracks to draw
        colormap: Optional dictionary mapping track_id to color
        show_ids: Whether to show track IDs
        show_inactive: Whether to show inactive tracks
        history_frames: Number of frames to show in trajectory
    
    Returns:
        Image with drawn tracks
    """
    # Create a copy of the image for drawing
    draw_img = image.copy()
    height, width = draw_img.shape[:2]
    
    # Initialize colormap if not provided
    if colormap is None:
        colormap = {}
    
    # Draw each track
    for track in tracks:
        # Skip inactive tracks if not showing them
        if not track.is_active and not show_inactive:
            continue
        
        # Assign color if not already in colormap
        if track.track_id not in colormap:
            # Generate a new color based on track ID (for consistency)
            hue = (track.track_id * 0.1) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR format
            colormap[track.track_id] = color
        
        # Get track color (black for inactive tracks)
        color = colormap[track.track_id]
        if not track.is_active:
            color = (128, 128, 128)  # Gray for inactive tracks
        
        # Get current position (most recent detection)
        if track.detections:
            det = track.detections[-1]
            x, y = int(det['x']), int(det['y'])
            w, h = int(det['width']), int(det['height'])
            
            # Draw bounding box
            cv2.rectangle(draw_img, (x, y), (x+w, y+h), color, 2)
            
            # Draw track ID if enabled
            if show_ids:
                text_pos = (x, y - 10) if y > 20 else (x, y + h + 20)
                cv2.putText(draw_img, f"ID:{track.track_id}", text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw trajectory (past positions)
        if len(track.detections) > 1:
            # Only show the last N frames in trajectory
            start_idx = max(0, len(track.detections) - history_frames)
            for i in range(start_idx + 1, len(track.detections)):
                prev_det = track.detections[i-1]
                curr_det = track.detections[i]
                
                prev_x = int(prev_det['x'] + prev_det['width']/2)
                prev_y = int(prev_det['y'] + prev_det['height']/2)
                curr_x = int(curr_det['x'] + curr_det['width']/2)
                curr_y = int(curr_det['y'] + curr_det['height']/2)
                
                # Draw connecting line for trajectory
                cv2.line(draw_img, (prev_x, prev_y), (curr_x, curr_y), color, 1)
    
    return draw_img


def visualize_tracks(sequence, tracks_by_frame, output_dir, history_frames=20):
    """
    Create visualizations of tracked birds.
    
    Args:
        sequence: List of image paths
        tracks_by_frame: Dictionary mapping frame_id to list of track dictionaries
        output_dir: Directory to save visualizations
        history_frames: Number of frames to show in trajectory
    """
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, "deep_visualized")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Prepare colormap for consistent colors
    colormap = {}
    
    # Create a dictionary to gather track information by track_id
    tracks_data = {}
    
    # First pass: gather track data by track_id
    for frame_id, tracks in tracks_by_frame.items():
        for track_dict in tracks:
            track_id = track_dict['track_id']
            
            if track_id not in tracks_data:
                tracks_data[track_id] = {
                    'frames': [],
                    'detections': []
                }
                
            tracks_data[track_id]['frames'].append(frame_id)
            tracks_data[track_id]['detections'].append(track_dict)
    
    # Process each frame
    for frame_id in tqdm(range(len(sequence)), desc="Creating visualizations"):
        # Skip if no tracks for this frame
        if frame_id not in tracks_by_frame:
            continue
            
        # Load image
        image_path = sequence[frame_id]
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Create a copy of the image for drawing
        viz_image = image.copy()
        
        # Get tracks for this frame
        current_tracks = tracks_by_frame[frame_id]
        
        # Draw each track with its history
        for track_dict in current_tracks:
            track_id = track_dict['track_id']
            
            # Skip if track data not found
            if track_id not in tracks_data:
                continue
                
            # Get track data
            track_data = tracks_data[track_id]
            frames = track_data['frames']
            detections = track_data['detections']
            
            # Assign color if not already in colormap
            if track_id not in colormap:
                # Generate a new color based on track ID (for consistency)
                hue = (track_id * 0.1) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
                color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR format
                colormap[track_id] = color
            
            # Get track color
            color = colormap[track_id]
            
            # Draw current bounding box
            x, y = int(track_dict['x']), int(track_dict['y'])
            w, h = int(track_dict['width']), int(track_dict['height'])
            cv2.rectangle(viz_image, (x, y), (x+w, y+h), color, 2)
            
            # Draw track ID
            cv2.putText(viz_image, f"ID:{track_id}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory (past positions)
            # Find the indices of frames in the history within the desired frame range
            frame_indices = [i for i, f in enumerate(frames) if f <= frame_id and f > frame_id - history_frames]
            
            # Draw lines between consecutive positions
            for i in range(1, len(frame_indices)):
                idx1 = frame_indices[i-1]
                idx2 = frame_indices[i]
                
                det1 = detections[idx1]
                det2 = detections[idx2]
                
                # Calculate center points
                x1 = int(det1['x'] + det1['width']/2)
                y1 = int(det1['y'] + det1['height']/2)
                x2 = int(det2['x'] + det2['width']/2)
                y2 = int(det2['y'] + det2['height']/2)
                
                # Draw connecting line
                cv2.line(viz_image, (x1, y1), (x2, y2), color, 2)
        
        # Save visualization
        output_path = os.path.join(viz_dir, f"deep_track_{frame_id:04d}.jpg")
        cv2.imwrite(output_path, viz_image)
    
    print(f"Visualizations saved to {viz_dir}")

def main():
    """Main function to run the deep learning enhanced bird tracker."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image sequence
    sequence = load_images(args.images_dir)
    print(f"Found {len(sequence)} images in sequence")
    
    # Load detections
    detections_by_frame = load_detections(args.detections_csv)
    print(f"Loaded detections from {len(detections_by_frame)} frames")
    
    # Start timer
    start_time = time.time()
    
    # Track birds
    print("Starting deep learning bird tracking...")
    tracks_by_frame, all_tracks = track_birds(
        sequence,
        detections_by_frame,
        max_distance=args.max_distance,
        iou_threshold=args.iou_threshold,
        appearance_weight=args.appearance_weight,
        deep_feature_weight=args.deep_feature_weight,
        inactive_frames=args.inactive_frames,
        min_hits=args.min_hits,
        use_deep_features=args.use_deep_features,
        feature_cache_size=args.feature_cache_size
    )
    
    # End timer
    elapsed_time = time.time() - start_time
    print(f"Tracking completed in {elapsed_time:.2f} seconds")
    
    # Analyze tracks
    stats = analyze_tracks(all_tracks)
    print(f"Total tracks: {stats['total_tracks']}")
    print(f"Average track length: {stats['avg_length']:.2f} detections")
    print(f"Average track duration: {stats['avg_duration']:.2f} frames")
    print(f"Longest track (ID {stats['longest_track_id']}): {stats['longest_track']} detections")
    print(f"Longest duration (ID {stats['longest_duration_id']}): {stats['longest_duration']} frames")
    
    # Save results to CSV
    output_csv = os.path.join(args.output_dir, "deep_tracked_detections.csv")
    total_entries = save_tracks_to_csv(tracks_by_frame, output_csv)
    print(f"Saved {total_entries} entries to {output_csv}")
    
    # Visualize tracks
    print("Creating track visualizations...")
    visualize_tracks(sequence, tracks_by_frame, args.output_dir, args.history_frames)
    
    # Clean up
    global feature_cache, frame_cache, model
    feature_cache = {}
    frame_cache = {}
    model = None
    
    print("Deep learning bird tracking completed!")


if __name__ == "__main__":
    main() 