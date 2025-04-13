import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from scipy.io import savemat
from multiprocessing import Pool

DATA_FOLDER = sys.argv[1]
POSE_FOLDER = "Poses"

os.environ['GLOG_minloglevel'] = '2'

def process_video(video_name):
    video_path = os.path.join(DATA_FOLDER, video_name)
    if not os.path.isdir(video_path):
        return

    frame_files = sorted(f for f in os.listdir(video_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    all_landmarks = []

    # IMPORTANT: create MediaPipe Pose INSIDE the subprocess
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        i = 0
        for frame in frame_files:
            i += 1
            if i % 2 != 0:
                continue
            frame_path = os.path.join(video_path, frame)
            image = cv2.imread(frame_path)
            if image is None:
                continue

            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            else:
                continue

            all_landmarks.append(frame_data)

    if all_landmarks:
        landmarks_array = np.stack(all_landmarks)
        os.makedirs(POSE_FOLDER, exist_ok=True)
        save_path = os.path.join(POSE_FOLDER, f'{video_name}.mat')
        savemat(save_path, {video_name: landmarks_array})
        print(f"Saved: {save_path} with shape {landmarks_array.shape}")


if __name__ == '__main__':
    # Run in parallel using all available cores
    video_list = os.listdir(DATA_FOLDER)
    with Pool() as pool:
        pool.map(process_video, video_list)
