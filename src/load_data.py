import kagglehub
from pathlib import Path
import process_video
import os
import shutil

DATASET_NAME = "dardarman1/golfing-posture"
SUBPATH = "CUSTOM DATASET/SIDE VIEW"
FRAMES_OUTPUT = "Data"

def download_data() -> str:
    path = kagglehub.dataset_download(DATASET_NAME)
    return path

def process_data(data_path: str) -> None:
    for root, _, files in os.walk(data_path):
        for file in files:
            bw_video = process_video.convert_video_to_grayscale(root, file)

            if bw_video is not None:
                relative_path = os.path.relpath(root, data_path)
                output_path = Path(FRAMES_OUTPUT) / Path(relative_path) / file[:-4]
                process_video.extract_frames(bw_video, output_path)

def run() -> None:
    full_dataset_path = download_data()
    data_path = Path(full_dataset_path) / Path(SUBPATH)
    process_data(data_path)
    shutil.rmtree(full_dataset_path)

if __name__ == "__main__":
    run()