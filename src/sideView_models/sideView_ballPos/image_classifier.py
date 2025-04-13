import cv2
import time
import joblib
import mediapipe as mp
import numpy as np
import logging
import argparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Classify golf ball position from live camera or image file.')
parser.add_argument('input_source', type=str, nargs='?', default='camera',
                    help='Input source: path to an image file or the keyword "camera" (default). ')
args = parser.parse_args()

# --- Configuration ---
MODEL_PATH = "swing_classifier.joblib"
ENCODER_PATH = "label_encoder.joblib"
COUNTDOWN_SECONDS = 5
CAPTURE_KEY = ord(' ')  # Spacebar
QUIT_KEY = ord('q')

# --- Mediapipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles # Added for styles

# --- Ball Detection Function (copied from extract_features.py) ---
def detect_ball(img):
    """Detects a golf ball using Hough Circle Transform."""
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5) # Use median blur for salt-and-pepper noise
    
    # --- HoughCircles Parameters (Tune these!) ---
    min_dist_pixels = int(height * 0.1) 
    canny_thresh = 100 
    accumulator_thresh = 20 # Lowered from 30 based on previous run
    min_radius_pixels = int(height * 0.01)
    max_radius_pixels = int(height * 0.05)
    # ---------------------------------------------

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, 
                               minDist=min_dist_pixels, 
                               param1=canny_thresh, param2=accumulator_thresh, 
                               minRadius=min_radius_pixels, maxRadius=max_radius_pixels)
    
    detected_ball_info = None
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Simple strategy: return the first detected circle
        # More robust: choose lowest circle or closest to center?
        if len(circles) > 0:
            ball_x, ball_y, ball_r = circles[0]
            logging.debug(f"Ball detected at ({ball_x}, {ball_y}) with radius {ball_r}")
            detected_ball_info = (ball_x, ball_y, ball_r)
    # Removed else logging for brevity if no ball detected
        
    # Return None or the detected circle info
    return detected_ball_info 

# --- Feature Extraction Function (updated to return pose_results) ---
def extract_features_from_live_frame(img):
    logging.debug("Attempting to extract features from live frame.")
    height, width, _ = img.shape
    features = None # Initialize features as None
    pose_results = None # Initialize pose_results as None

    # 1. Detect Ball first
    ball_info = detect_ball(img)
    
    # 2. Detect Pose (always attempt for visualization)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(img_rgb) # Store results here

    # 3. Try to calculate features IF ball AND pose landmarks are detected
    if ball_info is not None and pose_results and pose_results.pose_landmarks:
        logging.debug("Ball and Pose landmarks detected, attempting feature calculation.")
        ball_x_px, ball_y_px, ball_r_px = ball_info
        lm = pose_results.pose_landmarks.landmark
        
        try:
            left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate Features
            pelvis_x = (left_hip.x + right_hip.x) / 2
            left_x, right_x = left_ankle.x, right_ankle.x # Normalized coordinates (0.0 to 1.0)
            center_x = (left_x + right_x) / 2 # Center between ankles (normalized)
            stance_width = abs(left_x - right_x) + 1e-6  # Normalized stance width

            # Normalize detected ball position (pixel coord to relative coord 0.0-1.0)
            ball_x_norm = ball_x_px / width 

            # Calculate features relative to stance center and width
            features = {
                "left_ankle_x": (left_ankle.x - center_x) / stance_width,
                "right_ankle_x": (right_ankle.x - center_x) / stance_width,
                "pelvis_x": (pelvis_x - center_x) / stance_width,
                "ball_x": (ball_x_norm - center_x) / stance_width # Ball position relative to stance
            }
            logging.info(f"Successfully extracted features: {features}")

        except IndexError:
            logging.warning("Required landmarks (ankles/hips) not found, skipping feature calculation.")
            # Features remain None
        except Exception as e:
            logging.error(f"Error during feature calculation: {e}")
            # Features remain None
            
    elif ball_info is None:
        logging.warning("No ball detected, cannot calculate features.")
    elif not (pose_results and pose_results.pose_landmarks):
         logging.warning("Pose landmarks not detected, cannot calculate features.")
        
    # Return features (or None), ball_info (or None), and the pose results object (or None)
    return features, ball_info, pose_results 

# --- Load Model and Encoder ---
logging.info(f"Loading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: Model file not found at {MODEL_PATH}")
    exit()
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit()

logging.info(f"Loading label encoder from {ENCODER_PATH}...")
try:
    le = joblib.load(ENCODER_PATH)
    logging.info("Label encoder loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: Label encoder file not found at {ENCODER_PATH}")
    exit()
except Exception as e:
    logging.error(f"Error loading label encoder: {e}")
    exit()

# --- Input Processing ---
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

        cv2.imshow('Live Swing Analysis', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == CAPTURE_KEY:
            logging.info("Capture key pressed. Starting countdown...")
            for i in range(COUNTDOWN_SECONDS, 0, -1):
                start_time = time.time()
                while time.time() - start_time < 1.0: # Display each number for 1 second
                    ret, frame = cap.read()
                    if not ret: break
                    countdown_frame = frame.copy()
                    # Display countdown
                    text_size = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]
                    text_x = (countdown_frame.shape[1] - text_size[0]) // 2
                    text_y = (countdown_frame.shape[0] + text_size[1]) // 2
                    cv2.putText(countdown_frame, str(i), (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                    cv2.imshow('Live Swing Analysis', countdown_frame)
                    if cv2.waitKey(1) & 0xFF == QUIT_KEY: # Allow quitting during countdown
                       cap.release()
                       cv2.destroyAllWindows()
                       logging.info("Quit during countdown.")
                       exit()
                if not ret: break # Exit outer loop if frame read failed
            if not ret: break

            # Capture the frame *after* the countdown
            ret, captured_frame = cap.read()
            if ret:
                logging.info("Frame captured successfully.")
                break # Exit the loop after successful capture
            else:
                logging.error("Failed to capture frame after countdown.")
                break

        elif key == QUIT_KEY:
            logging.info("Quit key pressed. Exiting.")
            break

    # --- Cleanup Camera ---
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
    logging.info("Processing frame...")
    # Now returns features, ball info, AND pose results
    features_dict, detected_ball, pose_results = extract_features_from_live_frame(captured_frame)
    
    # Create a copy for drawing annotations
    annotated_frame = captured_frame.copy()
    height, width, _ = annotated_frame.shape # Get dimensions for coordinate conversion

    # Visualize detected ball (if any)
    if detected_ball:
        bx, by, br = detected_ball
        cv2.circle(annotated_frame, (bx, by), br, (0, 255, 0), 2) # Draw green circle
        cv2.circle(annotated_frame, (bx, by), 2, (0, 0, 255), 3) # Draw red center dot
        logging.info(f"Visualized ball at: ({bx}, {by}) R={br}")

    # Visualize detected pose (if any)
    pose_landmarks_present = pose_results and pose_results.pose_landmarks
    if pose_landmarks_present:
        logging.info("Visualizing detected pose landmarks.")
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Explicitly highlight key landmarks used in features
        key_landmarks_to_draw = {
            mp_pose.PoseLandmark.LEFT_ANKLE: "L Ankle",
            mp_pose.PoseLandmark.RIGHT_ANKLE: "R Ankle",
            mp_pose.PoseLandmark.LEFT_HIP: "L Hip",
            mp_pose.PoseLandmark.RIGHT_HIP: "R Hip"
        }
        lm = pose_results.pose_landmarks.landmark
        for landmark_id, name in key_landmarks_to_draw.items():
             # Check visibility if needed: if lm[landmark_id].visibility > 0.5:
             cx, cy = int(lm[landmark_id].x * width), int(lm[landmark_id].y * height)
             cv2.circle(annotated_frame, (cx, cy), 7, (255, 100, 0), cv2.FILLED) # Blue filled circle
             # cv2.putText(annotated_frame, name, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1) # Optional text label
    
    # --- Prediction (only if features were successfully extracted) ---
    prediction_text = "Prediction: N/A"
    if features_dict:
        logging.info("Making prediction...")
        # Ensure feature order matches training
        feature_order = ["left_ankle_x", "right_ankle_x", "pelvis_x", "ball_x"]
        try:
            features_for_prediction = np.array([[features_dict[feat] for feat in feature_order]])
            logging.debug(f"Features prepared for prediction: {features_for_prediction}")
            prediction_encoded = model.predict(features_for_prediction)
            prediction_proba = model.predict_proba(features_for_prediction)
            prediction_label = le.inverse_transform(prediction_encoded)[0]
            prediction_text = f"Prediction: {prediction_label}"
            
            logging.info("Prediction complete.")
            print("\n-------------------------------------")
            print(f"Predicted Ball Position: {prediction_label}")
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
        # Handle cases where features couldn't be extracted
        print("\n-------------------------------------")
        if detected_ball:
            print("Ball detected, but could not extract pose features. Prediction failed.")
            # Show the frame with just the detected ball
            cv2.imshow("Captured Frame Analysis (Ball Only)", annotated_frame)
            print("\nPress any key to close the result window.")
            cv2.waitKey(0)
            cv.destroyAllWindows()
        else:
             print("Could not detect ball or pose features from the captured frame. Prediction failed.")
        print("-------------------------------------")

    # Display the annotated frame with the prediction
    cv2.putText(annotated_frame, prediction_text, (10, annotated_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Captured Frame Analysis", annotated_frame)
    print("\nPress any key to close the result window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
     print("\nNo frame was captured.")

logging.info("Script finished.") 