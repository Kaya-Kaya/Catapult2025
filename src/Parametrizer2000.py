import kagglehub
import pandas as pd
import os
import cv2
import mediapipe as mp
import numpy as np
path = kagglehub.dataset_download("rakshitgirish/golf-pose")

def print_directory_tree(startpath, indent=''):
    for item in sorted(os.listdir(startpath)):
        full_path = os.path.join(startpath, item)
        if os.path.isdir(full_path):
            print(f"{indent}📁 {item}/")
            print_directory_tree(full_path, indent + '    ')
        else:
            print(f"{indent}📄 {item}")


def convert_videos_to_grayscale(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp4'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, f"bw_{file}")

                print(f"Processing: {input_path}")

                # Open the video file
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    print(f"❌ Failed to open {input_path}")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change to 'XVID' or 'avc1'
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out.write(gray_frame)

                cap.release()
                out.release()
                print(f"✅ Saved grayscale video to: {output_path}\n")

print("\n📂 Dataset Directory Tree:\n")
print_directory_tree(path)
convert_videos_to_grayscale(path)