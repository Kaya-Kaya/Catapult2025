import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
from scipy.io import savemat

DATA_FOLDER = sys.argv[1]
POSE_FOLDER = "Poses"

mp_pose = mp.solutions.pose

with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    for video_name in os.listdir(DATA_FOLDER):
        video_path = os.path.join(DATA_FOLDER, video_name)
        if not os.path.isdir(video_path):
            continue

        frame_files = sorted(os.listdir(video_path)) 
        all_landmarks = []
    
        for frame in frame_files:
            frame_path = os.path.join(video_path, frame)
        
            image = cv2.imread(frame_path)
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            else:
                frame_data = np.full((33, 4), np.nan)

            all_landmarks.append(frame_data)

        if all_landmarks:
            landmarks_array = np.stack(all_landmarks) # Shape: (# of frames, 33, 4)
            os.makedirs(POSE_FOLDER, exist_ok=True)
            save_path = os.path.join(POSE_FOLDER, f'{video_name}.mat')
            savemat(save_path, {video_name: landmarks_array})
            print(f"Saved: {video_name}.mat with shape {landmarks_array.shape}")

