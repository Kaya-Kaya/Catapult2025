import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from typing import List

DATA_FOLDER = "Data/Bad Swings/Bad Driver Swings"
POSE_FOLDER = "Poses"

def extract_poses(image_files: List[str]) -> None:
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:

        for idx, file in enumerate(image_files):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            
            # Prepare CSV file to write landmarks
            csv_directory = os.path.join(POSE_FOLDER, file[file.index("/") + 1:file.rindex("/")])
            os.makedirs(csv_directory, exist_ok=True)
            csv_file = os.path.join(csv_directory, file[file.rindex("/") + 1:-4]) + ".csv"
            with open(csv_file, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['Landmark', 'X', 'Y', 'Z', 'Visibility'])  # Header row

                # Write each landmark's data
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    csv_writer.writerow([i, landmark.x, landmark.y, landmark.z, landmark.visibility])


def run():
    image_files = []
    for root, _, files in os.walk(DATA_FOLDER):
        for file in files:
            relative_path = os.path.join(root, file)
            image_files.append(relative_path)
    
    extract_poses(image_files)
    

if __name__ == "__main__":
    run()