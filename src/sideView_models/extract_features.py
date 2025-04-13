import cv2
import os
import pandas as pd
from tqdm import tqdm
import logging
import mediapipe as mp
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIRS = [
    "output_frames/SIDE VIEW/Good Swings/Good Ball Position",
    "output_frames/SIDE VIEW/Bad Swings/Bad Ball Position Too Near",
    "output_frames/SIDE VIEW/Bad Swings/Bad Ball Position Too Far"
]

OUTPUT_CSV = "ball_position_features.csv"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_ball(img):
    """Detects a golf ball using Hough Circle Transform."""
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    min_dist_pixels = int(height * 0.1)
    canny_thresh = 100
    accumulator_thresh = 30
    min_radius_pixels = int(height * 0.01)
    max_radius_pixels = int(height * 0.05)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, 
                               minDist=min_dist_pixels, 
                               param1=canny_thresh, param2=accumulator_thresh, 
                               minRadius=min_radius_pixels, maxRadius=max_radius_pixels)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        ball_x, ball_y, ball_r = circles[0]
        logging.debug(f"Ball detected at ({ball_x}, {ball_y}) with radius {ball_r}")
        return ball_x, ball_y, ball_r
    else:
        logging.debug("No ball detected in frame.")
        return None, None, None

def extract_features_from_frame(img):
    height, width, _ = img.shape
    
    ball_x_px, _, _ = detect_ball(img)
    if ball_x_px is None:
        return None
        
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if not results.pose_landmarks:
            logging.warning("Pose not detected, skipping frame.")
            return None
        
        lm = results.pose_landmarks.landmark
        
        try:
            left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        except IndexError:
            logging.warning("Required landmarks (ankles/hips) not found, skipping frame.")
            return None

        pelvis_x = (left_hip.x + right_hip.x) / 2
        left_x, right_x = left_ankle.x, right_ankle.x
        center_x = (left_x + right_x) / 2
        stance_width = abs(left_x - right_x) + 1e-6

        ball_x_norm = ball_x_px / width 

        features = {
            "left_ankle_x": (left_ankle.x - center_x) / stance_width,
            "right_ankle_x": (right_ankle.x - center_x) / stance_width,
            "pelvis_x": (pelvis_x - center_x) / stance_width,
            "ball_x": (ball_x_norm - center_x) / stance_width
        }
        logging.debug(f"Extracted features: {features}")
        return features

def process_all():
    data = []
    frames_processed = 0
    frames_skipped_no_ball = 0
    frames_skipped_no_pose = 0
    
    for dir_path in INPUT_DIRS:
        logging.info(f"Processing directory: {dir_path}")
        video_folders = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        for video_folder in tqdm(video_folders, desc=f"Videos in {os.path.basename(dir_path)}"):
            video_path = os.path.join(dir_path, video_folder)

            frame_files = [f for f in os.listdir(video_path) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
            for frame_name in frame_files:
                frame_path = os.path.join(video_path, frame_name)
                img = cv2.imread(frame_path)
                if img is None:
                    logging.warning(f"Could not read frame: {frame_path}")
                    continue
                
                frames_processed += 1
                features = extract_features_from_frame(img)
                
                if features:
                    features["video"] = video_folder
                    features["frame"] = frame_name
                    features["label"] = os.path.basename(dir_path)
                    data.append(features)
                else:
                    if detect_ball(img)[0] is None:
                         frames_skipped_no_ball += 1
                    else: 
                         frames_skipped_no_pose += 1

    logging.info(f"Finished processing.")
    logging.info(f"Total frames processed: {frames_processed}")
    logging.info(f"Frames skipped (no ball detected): {frames_skipped_no_ball}")
    logging.info(f"Frames skipped (no pose detected): {frames_skipped_no_pose}")
    logging.info(f"Frames with features extracted: {len(data)}")

    if not data:
        logging.warning("No features were extracted. Check HoughCircle parameters or input data.")
        return
        
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved updated features to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_all()
