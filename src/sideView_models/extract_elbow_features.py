import cv2
import os
import pandas as pd
from tqdm import tqdm
import logging
import mediapipe as mp
import numpy as np
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Directories containing the *extracted frames*
INPUT_FRAME_DIRS = [
    "output_frames/SIDE VIEW/Good Swings/Good Elbow Posture Backswing",
    "output_frames/SIDE VIEW/Bad Swings/Bad Elbow Posture Backswing"
]

OUTPUT_CSV = "sideView_models/sideView_elbowBackswing/elbow_posture_features.csv"

# MediaPipe setup
mp_pose = mp.solutions.pose

# --- Helper Function: Calculate Angle ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points (angle at vertex b)."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    ba = a - b
    bc = c - b
    
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # Avoid division by zero and invalid arccos input
    if norm_ba == 0 or norm_bc == 0:
        return 0.0 
    cosine_angle = dot_product / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Ensure value is within [-1, 1]
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# --- Feature Extraction Function ---
def extract_elbow_features_from_frame(img):
    """Extracts elbow angle and upper arm angle from a frame."""
    
    features = None
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                # Get landmarks for the right arm (assuming side view)
                shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                hip = lm[mp_pose.PoseLandmark.RIGHT_HIP] # For torso reference
                
                # Check visibility (optional but recommended)
                if shoulder.visibility < 0.5 or elbow.visibility < 0.5 or wrist.visibility < 0.5 or hip.visibility < 0.5:
                    logging.debug("Key landmarks not sufficiently visible, skipping feature extraction.")
                    return None
                    
                # Calculate features
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                upper_arm_torso_angle = calculate_angle(hip, shoulder, elbow)
                
                features = {
                    "elbow_angle": elbow_angle,
                    "upper_arm_torso_angle": upper_arm_torso_angle
                }
                logging.debug(f"Extracted elbow features: {features}")
                
            except IndexError:
                logging.debug("Required pose landmarks (shoulder, elbow, wrist, hip) not found.")
            except Exception as e:
                 logging.error(f"Error calculating features: {e}")
        else:
             logging.debug("No pose landmarks detected in frame.")
             
    return features

# --- Process All Frames ---
def process_all_frames():
    all_features_data = []
    frames_processed = 0
    frames_skipped = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    for frame_base_dir in INPUT_FRAME_DIRS:
        logging.info(f"Processing frames directory: {frame_base_dir}")
        label = os.path.basename(frame_base_dir) # Get label from directory name
        
        if not os.path.isdir(frame_base_dir):
            logging.warning(f"Frame directory not found: {frame_base_dir}. Skipping.")
            continue
            
        video_frame_folders = [d for d in os.listdir(frame_base_dir) if os.path.isdir(os.path.join(frame_base_dir, d))]
        
        if not video_frame_folders:
             logging.warning(f"No video frame subfolders found in {frame_base_dir}. Skipping.")
             continue
             
        for video_folder in tqdm(video_frame_folders, desc=f"Videos in {label}"):
            video_path = os.path.join(frame_base_dir, video_folder)
            frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
            
            for frame_name in frame_files:
                frame_path = os.path.join(video_path, frame_name)
                img = cv2.imread(frame_path)
                if img is None:
                    logging.warning(f"Could not read frame: {frame_path}")
                    frames_skipped += 1
                    continue
                
                frames_processed += 1
                features = extract_elbow_features_from_frame(img)
                
                if features:
                    features["video"] = video_folder
                    features["frame"] = frame_name
                    features["label"] = label
                    all_features_data.append(features)
                else:
                    frames_skipped += 1

    logging.info(f"Finished processing all frame directories.")
    logging.info(f"Total frames processed: {frames_processed}")
    logging.info(f"Frames skipped (no features extracted): {frames_skipped}")
    logging.info(f"Frames with features extracted: {len(all_features_data)}")

    if not all_features_data:
        logging.warning("No features were extracted. Check input data or MediaPipe detection.")
        return
        
    df = pd.DataFrame(all_features_data)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved elbow posture features to {OUTPUT_CSV}")

# --- Main Execution ---
if __name__ == "__main__":
    process_all_frames() 