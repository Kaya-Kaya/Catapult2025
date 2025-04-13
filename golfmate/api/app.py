import os
import shutil
import cv2
import numpy as np
from scipy.io import loadmat, savemat
from flask import Flask, request, jsonify
import mediapipe as mp
import torch
from torch import nn

# Define the model class
class PoseScoringModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PoseScoringModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)
        out, _ = self.gru(X, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
FRAME_FOLDER = 'Frames'
POSE_FOLDER = 'Poses'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(POSE_FOLDER, exist_ok=True)

# Model configuration
MODEL_PATH = './model/model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 99   # 33 landmarks × 3 (x, y, z)
HIDDEN_SIZE = 64  # From checkpoint
NUM_LAYERS = 2    # From checkpoint
NUM_CLASSES = 9   # Matches the 9 motion metrics

# Load the model
model = PoseScoringModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def convert_to_grayscale(input_path, output_path):
    """Convert video to grayscale."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray_frame)
    
    cap.release()
    out.release()
    return output_path

def extract_frames(video_path, output_dir):
    """Extract frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
    cap.release()
    return count

def process_poses(frames_dir, output_path):
    """Process frames to extract pose landmarks and save as .mat file."""
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )
    
    all_landmarks = []
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        image = cv2.imread(frame_path)
        if image is None:
            continue
            
        results = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            all_landmarks.append(frame_data)
    
    mp_pose.close()
    
    if all_landmarks:
        landmarks_array = np.stack(all_landmarks)
        savemat(output_path, {'poses': landmarks_array})
        return landmarks_array
    return None

def score_poses(mat_path):
    """Load .mat file and pass through the model to get scores."""
    # Load .mat file
    mat_data = loadmat(mat_path)
    poses = mat_data['poses']  # Shape: (frames, 33, 4)
    
    # Preprocess data: Use only x, y, z (exclude visibility)
    poses = poses[:, :, :3].reshape(poses.shape[0], -1)  # Shape: (frames, 33 * 3 = 99)
    
    # Convert to tensor
    poses_tensor = torch.tensor(poses, dtype=torch.float32).to(device)
    
    # Add batch dimension: (1, frames, 99)
    poses_tensor = poses_tensor.unsqueeze(0)
    
    # Pass through model
    with torch.no_grad():
        scores = model(poses_tensor)
        scores = torch.sigmoid(scores).cpu().numpy().flatten()  # Apply sigmoid for [0,1] scores
    
    # Map scores to the correct metric names
    metric_names = [
        "chin motion",
        "right arm motion",
        "left arm motion",
        "back motion",
        "pelvic motion",
        "right knee motion",
        "left knee motion",
        "right feet motion",
        "left feet motion"
    ]
    scores_dict = {name: float(score) for name, score in zip(metric_names, scores)}
    
    return scores_dict

@app.route('/api/analyze-swing', methods=['POST'])
def analyze_swing():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
            
        video = request.files['video']
        if not video.filename.lower().endswith('.mp4'):
            return jsonify({'error': 'Only .mp4 files are supported'}), 400
            
        # Save uploaded video
        video_filename = video.filename
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video.save(video_path)
        
        # Step 1: Convert to grayscale
        grayscale_filename = f"bw_{video_filename}"
        grayscale_path = os.path.join(UPLOAD_FOLDER, grayscale_filename)
        convert_to_grayscale(video_path, grayscale_path)
        
        # Step 2: Extract frames
        frames_dir = os.path.join(FRAME_FOLDER, os.path.splitext(video_filename)[0])
        frame_count = extract_frames(grayscale_path, frames_dir)
        
        # Step 3: Process poses and save to .mat file
        pose_filename = f"{os.path.splitext(video_filename)[0]}.mat"
        pose_path = os.path.join(POSE_FOLDER, pose_filename)
        poses_array = process_poses(frames_dir, pose_path)
        
        if poses_array is None:
            return jsonify({'error': 'No poses detected in the video'}), 400
        
        # Step 4: Score poses using the model
        scores = score_poses(pose_path)
        
        # Cleanup
        for path in [video_path, grayscale_path, frames_dir]:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
        
        return jsonify({
            'message': 'Video processed successfully',
            'grayscale_video': grayscale_filename,
            'frame_count': frame_count,
            'pose_file': pose_filename,
            'pose_shape': list(poses_array.shape),
            'scores': scores
        })
        
    except Exception as e:
        # Cleanup in case of error
        for path in [video_path, grayscale_path, frames_dir]:
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)