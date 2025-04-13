import cv2
import os
from tqdm import tqdm

["output_frames/SIDE VIEW/Good Swings/Good Ball Position", "output_frames/SIDE VIEW/Bad Swings/Bad Ball Position Too Near", "output_frames/SIDE VIEW/Bad Swings/Bad Ball Position Too Far"]

INPUT_DIR = "data/SIDE VIEW/Bad Swings/Bad Ball Position Too Near"
OUTPUT_DIR = "output_frames/SIDE VIEW/Bad Swings/Bad Ball Position Too Near"
FPS = 10  # frames per second to extract

def extract_frames(video_path, save_dir, fps=10):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(vid_fps // fps)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_name = f"{saved_count:04}.jpg"
            cv2.imwrite(os.path.join(save_dir, frame_name), frame)
            saved_count += 1
        frame_count += 1

    cap.release()


def process_all_videos(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(video_path, input_dir)
                save_dir = os.path.join(output_dir, rel_path.replace('.mp4','').replace('.mov','').replace('.avi',''))
                extract_frames(video_path, save_dir, fps=FPS)


if __name__ == "__main__":
    process_all_videos(INPUT_DIR, OUTPUT_DIR)
    print(f"Frames extracted and saved to {OUTPUT_DIR}")

