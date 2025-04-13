import os
import shutil
from flask import Flask, request, jsonify
from process_video import extract_frames
# Assume there's a function to get scores from neural network
# from your_neural_network import get_posture_scores

# Temporary placeholder
def get_posture_scores(frame_paths):
    return

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/analyze-swing', methods=['POST'])
def analyze_swing():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        video = request.files['video']
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        frames_dir = os.path.join(UPLOAD_FOLDER, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        extract_frames(video_path, frames_dir)

        frame_paths = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')]
        scores = get_posture_scores(frame_paths)

        os.remove(video_path)
        shutil.rmtree(frames_dir)

        return jsonify({'scores': scores})
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)