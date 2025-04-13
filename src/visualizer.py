import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import namedtuple
from mediapipe.framework.formats import landmark_pb2

def display_frame_pose(input_file: str):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    BG_COLOR = (192, 192, 192) # gray
    # Read pose data from CSV
    pose_data = pd.read_csv(input_file)

    landmarks = []

    # Iterate through each frame in the pose data
    for idx, row in pose_data.iterrows():
        # Extract pose landmarks and create NormalizedLandmarkList
        landmarks.append(
            landmark_pb2.NormalizedLandmark(
                x=row['X'], y=row['Y'], z=row['Z'], visibility=row['Visibility']
            )
        )

    landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
    
    mp_drawing.plot_landmarks(
        landmark_list, mp_pose.POSE_CONNECTIONS)
    
if __name__ == "__main__":
    display_frame_pose("Poses/Good Iron Swings - Back/Good Iron Swing - Back 1/frame_0000.csv")