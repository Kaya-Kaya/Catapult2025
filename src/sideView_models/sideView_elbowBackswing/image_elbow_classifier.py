import cv2
import time
import joblib
import mediapipe as mp
import numpy as np
import logging
import argparse
import os
import math # Added for angle calculation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Classify elbow posture from live camera or image file.')
parser.add_argument('input_source', type=str, nargs='?', default='camera',
                    help='Input source: path to an image file or the keyword "camera" (default). ')
args = parser.parse_args()

# --- Configuration ---
# Paths relative to the workspace root
MODEL_PATH = "sideView_models/sideView_elbowBackswing/elbow_swing_classifier.joblib"
ENCODER_PATH = "sideView_models/sideView_elbowBackswing/elbow_label_encoder.joblib"
COUNTDOWN_SECONDS = 5
CAPTURE_KEY = ord(' ')  # Spacebar
QUIT_KEY = ord('q')

# --- Mediapipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# --- Helper Function: Calculate Angle (Copied from extract_elbow_features.py) ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 3D points (angle at vertex b)."""
    # Check if landmarks have visibility attribute and if they are visible
    # landmarks = [a, b, c]
    # if any(not hasattr(lm, 'visibility') or lm.visibility < 0.5 for lm in landmarks):
    #     # logging.debug("One or more landmarks for angle calculation not visible.")
    #     return 0.0 # Or handle as appropriate

    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        # logging.warning("Zero vector encountered in angle calculation.")
        return 0.0 
        
    dot_product = np.dot(ba, bc)
    cosine_angle = dot_product / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Ensure value is within [-1, 1]
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# --- Feature Extraction Function (Adapted for Elbow) ---
def extract_features_from_live_frame(img):
    logging.debug("Attempting to extract elbow features from live frame.")
    features = None # Initialize features as None
    pose_results = None # Initialize pose_results as None

    # Detect Pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(img_rgb)

    # Try to calculate features IF pose landmarks are detected
    if pose_results and pose_results.pose_landmarks:
        logging.debug("Pose landmarks detected, attempting feature calculation.")
        lm = pose_results.pose_landmarks.landmark
        
        try:
            # Get landmarks for the right arm (assuming side view)
            shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            hip = lm[mp_pose.PoseLandmark.RIGHT_HIP] # For torso reference
            
            # Check visibility (optional but recommended)
            min_visibility = 0.5
            if shoulder.visibility < min_visibility or elbow.visibility < min_visibility or wrist.visibility < min_visibility or hip.visibility < min_visibility:
                logging.warning("Key landmarks not sufficiently visible, skipping feature extraction.")
            else:
                # Calculate features
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                upper_arm_torso_angle = calculate_angle(hip, shoulder, elbow)
                
                features = {
                    "elbow_angle": elbow_angle,
                    "upper_arm_torso_angle": upper_arm_torso_angle
                }
                logging.info(f"Successfully extracted elbow features: {features}")

        except IndexError:
            logging.warning("Required pose landmarks (shoulder, elbow, wrist, hip) not found.")
            # Features remain None
        except Exception as e:
            logging.error(f"Error during feature calculation: {e}")
            # Features remain None
            
    else:
         logging.warning("Pose landmarks not detected, cannot calculate features.")
        
    # Return features (or None) and the pose results object (or None)
    return features, pose_results 

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

# --- Input Processing (Camera or Image File) ---
captured_frame = None

if args.input_source.lower() == 'camera':
    logging.info("Initializing camera...")
    cap = cv2.VideoCapture(0) # 0 is the default camera
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        exit()
    logging.info("Camera opened. Press SPACEBAR to start capture countdown, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to grab frame.")
            break
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press SPACEBAR to Capture | 'q' to Quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Live Elbow Analysis', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == CAPTURE_KEY:
            logging.info("Capture key pressed. Starting countdown...")
            for i in range(COUNTDOWN_SECONDS, 0, -1):
                start_time = time.time()
                while time.time() - start_time < 1.0: # Display each number for 1 second
                    ret, frame = cap.read()
                    if not ret: break
                    countdown_frame = frame.copy()
                    text_size = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]
                    text_x = (countdown_frame.shape[1] - text_size[0]) // 2
                    text_y = (countdown_frame.shape[0] + text_size[1]) // 2
                    cv2.putText(countdown_frame, str(i), (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                    cv2.imshow('Live Elbow Analysis', countdown_frame)
                    if cv2.waitKey(1) & 0xFF == QUIT_KEY: # Allow quitting during countdown
                       cap.release()
                       cv2.destroyAllWindows()
                       logging.info("Quit during countdown.")
                       exit()
                if not ret: break
            if not ret: break
            ret, captured_frame = cap.read()
            if ret:
                logging.info("Frame captured successfully.")
                break 
            else:
                logging.error("Failed to capture frame after countdown.")
                break
        elif key == QUIT_KEY:
            logging.info("Quit key pressed. Exiting.")
            break
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Camera released and windows closed.")

elif os.path.exists(args.input_source):
    logging.info(f"Loading image from file: {args.input_source}")
    captured_frame = cv2.imread(args.input_source)
    if captured_frame is None:
        logging.error(f"Error: Could not read image file: {args.input_source}")
        exit()
    logging.info("Image loaded successfully.")
else:
    logging.error(f"Error: Input source not found or invalid: {args.input_source}")
    logging.info("Please provide a valid image path or the keyword 'camera'.")
    exit()

# --- Feature Extraction, Prediction, and Visualization --- 
if captured_frame is not None:
    logging.info("Processing frame for elbow features...")
    features_dict, pose_results = extract_features_from_live_frame(captured_frame)
    
    annotated_frame = captured_frame.copy()
    height, width, _ = annotated_frame.shape

    # Visualize detected pose (if any)
    pose_landmarks_present = pose_results and pose_results.pose_landmarks
    if pose_landmarks_present:
        logging.info("Visualizing detected pose landmarks.")
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Explicitly highlight key landmarks used for elbow features
        key_landmarks_to_draw = {
            mp_pose.PoseLandmark.RIGHT_SHOULDER: "R Shoulder",
            mp_pose.PoseLandmark.RIGHT_ELBOW: "R Elbow",
            mp_pose.PoseLandmark.RIGHT_WRIST: "R Wrist",
            mp_pose.PoseLandmark.RIGHT_HIP: "R Hip"
        }
        lm = pose_results.pose_landmarks.landmark
        for landmark_id, name in key_landmarks_to_draw.items():
            if int(landmark_id) < len(lm) and lm[landmark_id].visibility > 0.3: # Check visibility
                 cx, cy = int(lm[landmark_id].x * width), int(lm[landmark_id].y * height)
                 cv2.circle(annotated_frame, (cx, cy), 7, (255, 100, 0), cv2.FILLED) # Blue filled circle
    
    # --- Prediction (only if features were successfully extracted) ---
    prediction_text = "Prediction: N/A"
    if features_dict:
        logging.info("Making prediction...")
        feature_order = ["elbow_angle", "upper_arm_torso_angle"]
        try:
            features_for_prediction = np.array([[features_dict[feat] for feat in feature_order]])
            logging.debug(f"Features prepared for prediction: {features_for_prediction}")
            prediction_encoded = model.predict(features_for_prediction)
            prediction_proba = model.predict_proba(features_for_prediction)
            prediction_label = le.inverse_transform(prediction_encoded)[0]
            prediction_text = f"Prediction: {prediction_label}"
            
            logging.info("Prediction complete.")
            print("\n-------------------------------------")
            print(f"Predicted Elbow Posture: {prediction_label}")
            print("Prediction Probabilities:")
            for label, proba in zip(le.classes_, prediction_proba[0]):
                 print(f"  - {label}: {proba:.4f}")
            print("-------------------------------------")
        except KeyError as e:
             logging.error(f"Feature key missing for prediction: {e}")
             print(f"\nError: Feature key missing ({e}). Cannot predict.")
             prediction_text = "Prediction: Error"
        except Exception as e:
             logging.error(f"Error during prediction: {e}")
             print(f"\nError during prediction: {e}")
             prediction_text = "Prediction: Error"

    else:
        print("\n-------------------------------------")
        if pose_landmarks_present:
             print("Pose detected, but could not extract required features (check landmark visibility?).")
        else:
             print("Could not detect pose landmarks. Feature extraction and prediction failed.")
        print("-------------------------------------")

    # Display the annotated frame with the prediction
    cv2.putText(annotated_frame, prediction_text, (10, annotated_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Smaller font
    cv2.imshow("Live Elbow Analysis Result", annotated_frame)
    print("\nPress any key to close the result window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
     print("\nNo frame was captured or loaded.")

logging.info("Script finished.") 