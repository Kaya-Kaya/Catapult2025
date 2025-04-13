import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_CSV = "sideView_models/sideView_elbowBackswing/elbow_posture_features.csv"
OUTPUT_PLOT_FILE = "sideView_models/sideView_elbowBackswing/elbow_feature_distribution.png"

# --- Load Data ---
logging.info(f"Loading data from {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"Error: Input file not found at {INPUT_CSV}")
    exit()
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit()

# --- Visualization ---
logging.info("Generating feature distribution plot...")

plt.figure(figsize=(10, 8))

sns.scatterplot(data=df, x='elbow_angle', y='upper_arm_torso_angle', hue='label', 
                palette='viridis', alpha=0.7, s=50) # s is marker size

plt.title('Elbow Posture Feature Distribution', fontsize=16)
plt.xlabel('Elbow Angle (degrees)', fontsize=12)
plt.ylabel('Upper Arm-Torso Angle (degrees)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Posture Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

# --- Save Plot ---
try:
    plt.savefig(OUTPUT_PLOT_FILE)
    logging.info(f"Plot saved successfully to {OUTPUT_PLOT_FILE}")
except Exception as e:
    logging.error(f"Error saving plot: {e}")

# Optionally display the plot if running in an interactive environment
# plt.show() 

logging.info("Visualization script finished.") 