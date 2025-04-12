import kagglehub
from pathlib import Path
import process_video
import os

DATASET_NAME = "rakshitgirish/golf-pose"
SUBPATH = "CUSTOM DATASET/SIDE VIEW"

def download_data() -> str:
    path = kagglehub.dataset_download(DATASET_NAME)
    return path

def process_data(data_path: str):
    for root, _, files in os.walk(data_path):
        for file in files:
            process_video.convert_video_to_grayscale(root, file)

def run() -> None:
    data_path = Path(download_data()) / Path(SUBPATH)
    print(data_path)
    process_data(data_path)

if __name__ == "__main__":
    run()