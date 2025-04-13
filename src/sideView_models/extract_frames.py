import cv2
import os
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
FPS = 10  # frames per second to extract

# Define pairs of input video directories and output frame directories
# Paths are now relative to the WORKSPACE ROOT
DATA_DIRS = [
    {
        "input": "data/SIDE VIEW/Good Swings/Good Elbow Posture Backswing", 
        "output": "output_frames/SIDE VIEW/Good Swings/Good Elbow Posture Backswing"
    },
    {
        "input": "data/SIDE VIEW/Bad Swings/Bad Elbow Posture Backswing",
        "output": "output_frames/SIDE VIEW/Bad Swings/Bad Elbow Posture Backswing"
    }
]

def extract_frames(video_path, save_dir, fps=10):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Could not open video file: {video_path}")
        return
        
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if vid_fps is None or vid_fps <= 0:
        logging.warning(f"Could not get valid FPS from video: {video_path}. Skipping.")
        cap.release()
        return
        
    frame_interval = max(1, int(vid_fps // fps)) # Ensure interval is at least 1

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_name = f"{saved_count:04}.jpg"
            save_path = os.path.join(save_dir, frame_name)
            try:
                cv2.imwrite(save_path, frame)
                saved_count += 1
            except Exception as e:
                logging.error(f"Error writing frame {save_path}: {e}")
        frame_count += 1

    cap.release()
    # logging.info(f"Saved {saved_count} frames from {os.path.basename(video_path)} to {os.path.basename(save_dir)}")

def process_all_videos(input_dir, output_dir):
    logging.info(f"Processing videos from: {input_dir}")
    if not os.path.isdir(input_dir):
        logging.warning(f"Input directory not found: {input_dir}. Skipping.")
        return
        
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                 video_files.append(os.path.join(root, file))

    if not video_files:
        logging.warning(f"No video files found in {input_dir}.")
        return

    for video_path in tqdm(video_files, desc=f"Extracting frames in {os.path.basename(input_dir)}"):
        # Create a unique directory for each video's frames
        # Output path structure: output_dir / video_filename_without_ext /
        video_filename = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_filename)[0]
        save_dir = os.path.join(output_dir, video_name_no_ext) 
        
        extract_frames(video_path, save_dir, fps=FPS)

if __name__ == "__main__":
    logging.info("Starting frame extraction process for elbow posture...")
    for dir_pair in DATA_DIRS:
        input_video_dir = dir_pair["input"]
        output_frames_dir = dir_pair["output"]
        process_all_videos(input_video_dir, output_frames_dir)
        
    logging.info("Frame extraction process completed.")

