import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Adjust path relative to the workspace root where the script will likely be run from
INPUT_CSV = "sideView_models/sideView_elbowBackswing/elbow_posture_features.csv"
OUTPUT_MODEL_DIR = "sideView_models/sideView_elbowBackswing"
OUTPUT_MODEL_FILE = os.path.join(OUTPUT_MODEL_DIR, "elbow_swing_classifier.joblib")
OUTPUT_ENCODER_FILE = os.path.join(OUTPUT_MODEL_DIR, "elbow_label_encoder.joblib")
N_SPLITS = 1 # Number of shuffle splits to generate
TEST_SIZE = 0.2  # 20% of groups (videos) for testing
RANDOM_STATE = 42 # for reproducibility

# Ensure output directory exists
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# --- Load Data ---
logging.info(f"Loading data from {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    logging.debug(f"Data head:\n{df.head()}")
except FileNotFoundError:
    logging.error(f"Error: Input file not found at {INPUT_CSV}")
    exit()
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit()

# --- Feature, Target, and Group Selection ---
features = ["elbow_angle", "upper_arm_torso_angle"]
target = "label"
group_col = "video" # Column identifying the video each frame belongs to

logging.info(f"Selected features: {features}")
logging.info(f"Selected target: {target}")
logging.info(f"Selected group column: {group_col}")

X = df[features]
y_raw = df[target]
groups = df[group_col]

# --- Label Encoding ---
logging.info("Encoding target labels...")
le = LabelEncoder()
y = le.fit_transform(y_raw)
logging.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
logging.debug(f"Encoded labels preview: {y[:5]}")

# --- Group-Based Train/Test Split ---
logging.info(f"Splitting data based on groups ({group_col})...")
gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Get the indices for the first (and only, in this case) split
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

logging.info(f"Split complete.")
logging.info(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
logging.info(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
logging.info(f"Number of unique groups in training set: {groups_train.nunique()}")
logging.info(f"Number of unique groups in testing set: {groups_test.nunique()}")
# Verify no overlap in groups
train_groups = set(groups_train.unique())
test_groups = set(groups_test.unique())
assert train_groups.isdisjoint(test_groups), "Error: Groups overlap between train and test sets!"
logging.info("Group split verified: No overlap between train and test video groups.")

# --- Model Training ---
logging.info("Initializing Random Forest Classifier with OOB score enabled...")
model = RandomForestClassifier(n_estimators=100, 
                             random_state=RANDOM_STATE, 
                             class_weight='balanced',
                             oob_score=True, # Enable OOB score calculation
                             n_jobs=-1) # Use all available CPU cores for training
logging.info(f"Training model with parameters: {model.get_params()}")

model.fit(X_train, y_train)
logging.info("Model training completed.")

# Print the OOB score
if hasattr(model, 'oob_score_'):
    logging.info(f"Out-of-Bag (OOB) Score: {model.oob_score_:.4f}")
else:
     logging.warning("OOB score was not calculated (requires n_estimators >= 1 and bootstrap=True).")

# --- Evaluation ---
logging.info("Evaluating model on the test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

logging.info(f"Test Set Accuracy: {accuracy:.4f}")
logging.info(f"Classification Report:\n{report}")

# --- Save Model ---
logging.info(f"Saving trained elbow model to {OUTPUT_MODEL_FILE}...")
try:
    joblib.dump(model, OUTPUT_MODEL_FILE)
    joblib.dump(le, OUTPUT_ENCODER_FILE)
    logging.info("Model and label encoder saved successfully.")
except Exception as e:
    logging.error(f"Error saving model: {e}")

logging.info("Script finished.") 