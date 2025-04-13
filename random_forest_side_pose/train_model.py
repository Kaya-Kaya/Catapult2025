import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_CSV = "ball_position_features.csv"
OUTPUT_MODEL_FILE = "swing_classifier.joblib"
TEST_SIZE = 0.2  # 20% of data for testing
RANDOM_STATE = 42 # for reproducibility

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

# --- Feature and Target Selection ---
features = ["left_ankle_x", "right_ankle_x", "pelvis_x", "ball_x"]
target = "label"

logging.info(f"Selected features: {features}")
logging.info(f"Selected target: {target}")

X = df[features]
y_raw = df[target]

# --- Label Encoding ---
logging.info("Encoding target labels...")
le = LabelEncoder()
y = le.fit_transform(y_raw)
logging.info(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
logging.debug(f"Encoded labels preview: {y[:5]}")

# --- Train/Test Split ---
logging.info(f"Splitting data into training and testing sets (Test size: {TEST_SIZE}, Random State: {RANDOM_STATE})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y # Important for imbalanced datasets
)
logging.info(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
logging.info(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# --- Model Training ---
logging.info("Initializing Random Forest Classifier...")
# We can add hyperparameters here if needed, e.g., n_estimators, max_depth
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced') # class_weight='balanced' can help with imbalanced classes
logging.info(f"Training model with parameters: {model.get_params()}")

# NOTE: RandomForest itself doesn't have a built-in progress bar for training.
# For very large datasets, you might explore libraries like Dask or XGBoost which have better progress reporting.
# Since this dataset is likely small, training should be quick.
model.fit(X_train, y_train)
logging.info("Model training completed.")

# --- Evaluation ---
logging.info("Evaluating model on the test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

logging.info(f"Test Set Accuracy: {accuracy:.4f}")
logging.info(f"Classification Report:\n{report}")

# --- Save Model ---
logging.info(f"Saving trained model to {OUTPUT_MODEL_FILE}...")
try:
    joblib.dump(model, OUTPUT_MODEL_FILE)
    joblib.dump(le, 'label_encoder.joblib') # Save the encoder too!
    logging.info("Model and label encoder saved successfully.")
except Exception as e:
    logging.error(f"Error saving model: {e}")

logging.info("Script finished.")
