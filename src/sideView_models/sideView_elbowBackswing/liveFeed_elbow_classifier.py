import cv2
import time
import joblib
import mediapipe as mp
import numpy as np
import logging
import os
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_PATH = "sideView_models/sideView_elbowBackswing/elbow_swing_classifier.joblib"
ENCODER_PATH = "sideView_models/sideView_elbowBackswing/elbow_label_encoder.joblib"
TARGET_FPS = 10 # Target processing FPS (actual FPS might be lower due to processing time)
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5 # Added for non-static mode
QUIT_KEY = ord('q')

# --- Mediapipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# --- Helper Function: Calculate Angle ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points (angle at vertex b)."""
    # Visibility checks can be done before calling this if needed
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0 
    dot_product = np.dot(ba, bc)
    cosine_angle = dot_product / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# --- Feature Extraction Logic --- 
def extract_elbow_features(results):
    """Extracts elbow features if landmarks are present and visible."""
    features = None
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        try:
            shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            
            min_visibility = 0.5 # Visibility threshold
            if shoulder.visibility > min_visibility and elbow.visibility > min_visibility and wrist.visibility > min_visibility and hip.visibility > min_visibility:
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                upper_arm_torso_angle = calculate_angle(hip, shoulder, elbow)
                features = {
                    "elbow_angle": elbow_angle,
                    "upper_arm_torso_angle": upper_arm_torso_angle
                }
            else:
                logging.debug("Key landmarks not sufficiently visible.")
        except (IndexError, KeyError):
            logging.debug("Required pose landmarks not found in detected pose.")
        except Exception as e:
            logging.error(f"Error calculating features: {e}")
    return features

# --- Load Model and Encoder ---
logging.info(f"Loading elbow model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Elbow model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: Elbow model file not found at {MODEL_PATH}")
    exit()
except Exception as e:
    logging.error(f"Error loading elbow model: {e}")
    exit()

logging.info(f"Loading elbow label encoder from {ENCODER_PATH}...")
try:
    le = joblib.load(ENCODER_PATH)
    logging.info("Elbow label encoder loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: Elbow label encoder file not found at {ENCODER_PATH}")
    exit()
except Exception as e:
    logging.error(f"Error loading elbow label encoder: {e}")
    exit()

# --- Real-time Processing Setup ---
logging.info("Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Error: Could not open camera.")
    exit()

# Attempt to set camera FPS (might not be supported by all cameras/drivers)
try:
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Requested camera FPS: {TARGET_FPS}, Actual camera FPS reported: {actual_fps}")
except Exception as e:
    logging.warning(f"Could not set or get camera FPS: {e}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logging.info(f"Camera resolution: {frame_width}x{frame_height}")

prev_time = 0

# Initialize MediaPipe Pose for video stream
with mp_pose.Pose(
    static_image_mode=False, # Process as video stream
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF) as pose:

    logging.info("Starting real-time elbow analysis. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Ignoring empty camera frame.")
            continue

        # --- Performance measurement start ---
        current_time = time.time()
        
        # --- Image Processing ---
        # Convert the BGR image to RGB, flip for selfie view, process with MediaPipe Pose.
        frame.flags.writeable = False # Make immutable for performance
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        frame.flags.writeable = True # Make mutable again for drawing
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # Convert back if needed for drawing?
        
        # --- Feature Extraction & Prediction ---
        prediction_text = "Prediction: N/A"
        features = extract_elbow_features(results)
        
        if features:
            feature_order = ["elbow_angle", "upper_arm_torso_angle"]
            try:
                features_for_prediction = np.array([[features[feat] for feat in feature_order]])
                prediction_encoded = model.predict(features_for_prediction)
                # prediction_proba = model.predict_proba(features_for_prediction)
                prediction_label = le.inverse_transform(prediction_encoded)[0]
                prediction_text = f"Prediction: {prediction_label}"
                logging.debug(f"Prediction: {prediction_label}")
            except Exception as e:
                 logging.error(f"Prediction failed: {e}")
                 prediction_text = "Prediction: Error"

        # --- Visualization ---
        annotated_frame = frame # Draw directly on the frame
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Highlight key landmarks
            key_landmarks_to_draw = {
                mp_pose.PoseLandmark.RIGHT_SHOULDER: "R Shoulder",
                mp_pose.PoseLandmark.RIGHT_ELBOW: "R Elbow",
                mp_pose.PoseLandmark.RIGHT_WRIST: "R Wrist",
                mp_pose.PoseLandmark.RIGHT_HIP: "R Hip"
            }
            lm = results.pose_landmarks.landmark
            for landmark_id, name in key_landmarks_to_draw.items():
                if int(landmark_id) < len(lm) and lm[landmark_id].visibility > 0.3:
                     cx, cy = int(lm[landmark_id].x * frame_width), int(lm[landmark_id].y * frame_height)
                     cv2.circle(annotated_frame, (cx, cy), 7, (255, 100, 0), cv2.FILLED) # Blue filled circle
        
        # --- Display FPS and Prediction ---
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, prediction_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Real-time Elbow Analysis', annotated_frame)

        # Check for quit key
        if cv2.waitKey(5) & 0xFF == QUIT_KEY:
            logging.info("Quit key pressed. Exiting.")
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
logging.info("Script finished.")
